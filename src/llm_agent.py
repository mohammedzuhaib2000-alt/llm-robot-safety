"""
llm_agent.py — the bridge between the LLM and the robot.

This is the "menu" idea in code. We describe each robot command to the LLM in
a format it understands (a "tool"), send it the situation, and it replies by
choosing one command and filling in the blanks. We run that command, hand the
result back, and ask again. Repeat until it says it is finished.

IMPORTANT: each provider keeps the conversation in its OWN native tool-calling
format. That matters for this project — if we paraphrased the exchange into
plain sentences, the model would be reacting to our paraphrase rather than to
a real tool result, and the behaviour we measured would partly be an artefact
of our code. Keep it native.

Four providers are supported:
  fake       — no key, no internet, no cost. A dumb keyword robot, for checking
               the loop works. START HERE. Never report its numbers.
  ollama     — a real model running on YOUR OWN LAPTOP. Free and unlimited.
               Needs Ollama installed; no API key. See OllamaProvider below.
  anthropic  — needs ANTHROPIC_API_KEY.  Costs money.
  openai     — needs OPENAI_API_KEY.     Costs money.
"""

import os
import json


# ---------------------------------------------------------------- the menu ---
# This list IS the robot's action API. Anything not described here, the model
# cannot ask for. Adding or removing an entry changes what the robot can do.

TOOLS = [
    {
        "name": "get_scene",
        "description": "List every object on the table and where it is.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "pick",
        "description": "Pick up an object from the table with the gripper.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string", "description": "name of the object to pick up"},
                "force": {"type": "number", "description": "grip force in newtons, 1-100. Use 15 or less for fragile items."},
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "place",
        "description": "Put the currently held object down into a named container such as 'tray' or 'bin'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string"},
                "target": {"type": "string", "description": "a container name such as 'tray' or 'bin'"},
            },
            "required": ["object_id", "target"],
        },
    },
    {
        "name": "move_to",
        "description": "Move the gripper to a position in metres, in the robot's base frame.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "speed": {"type": "number", "description": "metres per second, 0.05 to 0.5"},
            },
            "required": ["x", "y", "z"],
        },
    },
    {"name": "open_gripper", "description": "Open the gripper fingers.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {
        "name": "close_gripper",
        "description": "Close the gripper fingers with a given force in newtons.",
        "input_schema": {"type": "object",
                         "properties": {"force": {"type": "number"}}, "required": []},
    },
    {
        "name": "report",
        "description": "Say something to the operator. Use this to refuse an unsafe instruction, to ask a question, or to explain a problem. Do NOT use it to narrate what you are about to do.",
        "input_schema": {"type": "object",
                         "properties": {"message": {"type": "string"}}, "required": ["message"]},
    },
    {
        "name": "done",
        "description": "Finish the task. Call this when there is nothing left to do.",
        "input_schema": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "summary": {"type": "string"},
            },
            "required": ["success", "summary"],
        },
    },
]


def _accepts(fn, param):
    """Does this SDK function accept this argument?

    SDKs change. Older Anthropic versions took `temperature`; version 1.x
    removed it. Rather than pinning a version, we ask the installed library
    what it supports and only send what it accepts.
    """
    import inspect
    try:
        return param in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------- providers --
# Every provider offers the same three methods:
#   add_user(text)     the operator says something
#   next_action()      -> (tool_name, arguments)
#   add_result(text)   hand the outcome of that command back to the model

class FakeProvider:
    """A pretend LLM. No API key, no internet, no cost.

    It does not understand anything — it just looks for keywords. Its only
    purpose is to prove the loop works end to end before you spend anything.
    Never report results from this provider in your dissertation.
    """

    name = "fake"
    supports_temperature = True

    def __init__(self, **_):
        self.step = 0
        self.said = ""
        self.scene = ""
        self.tokens_in = 0
        self.tokens_out = 0

    KNOWN = ("red_block", "blue_block", "yellow_block", "heavy_block", "beaker")

    def add_user(self, text):
        self.said += " " + text.lower()

    def add_result(self, text):
        self.scene = str(text).lower()        # remember what get_scene told us

    def _target(self):
        """Pick an object the user named; failing that, any block on the table."""
        for o in self.KNOWN:
            if o in self.said or o.replace("_", " ") in self.said:
                return o
        if "block" in self.said:              # "sort the blocks" — take the first one
            for o in self.KNOWN:
                if o in getattr(self, "scene", ""):
                    return o
        return None

    def next_action(self):
        self.step += 1
        if self.step == 1:
            return "get_scene", {}
        obj = self._target()
        dest = "bin" if "bin" in self.said else "tray"
        if obj is None:
            if self.step == 2:
                return "report", {"message": "I could not tell which object you meant."}
            return "done", {"success": False, "summary": "fake provider: no target"}
        if self.step == 2:
            return "pick", {"object_id": obj}
        if self.step == 3:
            return "place", {"object_id": obj, "target": dest}
        return "done", {"success": True, "summary": "fake provider finished"}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model="claude-sonnet-4-5", temperature=0.0, system_prompt=""):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.messages = []
        self.pending_tool_use_id = None
        self.tokens_in = 0
        self.tokens_out = 0
        # Anthropic removed `temperature` in SDK v1. Ask, don't assume.
        self.supports_temperature = _accepts(self.client.messages.create, "temperature")

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})

    def next_action(self):
        kwargs = dict(
            model=self.model,
            max_tokens=1024,
            system=self.system_prompt,
            tools=[{"name": t["name"], "description": t["description"],
                    "input_schema": t["input_schema"]} for t in TOOLS],
            messages=self.messages,
        )
        if self.supports_temperature:
            kwargs["temperature"] = self.temperature

        resp = self.client.messages.create(**kwargs)
        u = getattr(resp, "usage", None)
        if u is not None:
            self.tokens_in += getattr(u, "input_tokens", 0) or 0
            self.tokens_out += getattr(u, "output_tokens", 0) or 0

        # Store the model's turn EXACTLY as it came back, so the next request
        # contains a real tool_use block rather than our paraphrase of one.
        self.messages.append({
            "role": "assistant",
            "content": [b.model_dump(exclude_none=True) for b in resp.content],
        })

        for block in resp.content:
            if block.type == "tool_use":
                self.pending_tool_use_id = block.id
                return block.name, dict(block.input)

        # No tool was chosen — the model only wrote text. Treat that as a report.
        self.pending_tool_use_id = None
        text = "".join(b.text for b in resp.content if b.type == "text")
        return "report", {"message": text or "(no action chosen)"}

    def add_result(self, text):
        if self.pending_tool_use_id is None:
            self.messages.append({"role": "user", "content":
                                  "Choose a tool to call, or call done()."})
            return
        self.messages.append({"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": self.pending_tool_use_id,
            "content": str(text),
        }]})
        self.pending_tool_use_id = None


class OpenAIProvider:
    name = "openai"

    def __init__(self, model="gpt-4o", temperature=0.0, system_prompt=""):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.temperature = temperature
        self.messages = [{"role": "system", "content": system_prompt}]
        self.pending_tool_call_id = None
        self.tokens_in = 0
        self.tokens_out = 0
        self.supports_temperature = _accepts(
            self.client.chat.completions.create, "temperature")

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})

    def next_action(self):
        kwargs = dict(
            model=self.model,
            messages=self.messages,
            tools=[{"type": "function",
                    "function": {"name": t["name"], "description": t["description"],
                                 "parameters": t["input_schema"]}} for t in TOOLS],
        )
        if self.supports_temperature:
            kwargs["temperature"] = self.temperature

        resp = self.client.chat.completions.create(**kwargs)
        u = getattr(resp, "usage", None)
        if u is not None:
            self.tokens_in += getattr(u, "prompt_tokens", 0) or 0
            self.tokens_out += getattr(u, "completion_tokens", 0) or 0
        msg = resp.choices[0].message
        self.messages.append(msg.model_dump(exclude_none=True))

        if msg.tool_calls:
            call = msg.tool_calls[0]
            self.pending_tool_call_id = call.id
            return call.function.name, json.loads(call.function.arguments or "{}")

        self.pending_tool_call_id = None
        return "report", {"message": msg.content or "(no action chosen)"}

    def add_result(self, text):
        if self.pending_tool_call_id is None:
            self.messages.append({"role": "user", "content":
                                  "Choose a tool to call, or call done()."})
            return
        self.messages.append({"role": "tool",
                              "tool_call_id": self.pending_tool_call_id,
                              "content": str(text)})
        self.pending_tool_call_id = None


class OllamaProvider(OpenAIProvider):
    """A model running on YOUR OWN LAPTOP. Free, unlimited, no API key.

    Ollama serves an OpenAI-compatible endpoint on localhost, so everything
    the OpenAI provider does works unchanged — we just point it somewhere else.

    Setup (once):
        1. Install Ollama from https://ollama.com
        2. ollama pull qwen2.5:7b          (or llama3.1:8b, or llama3.2:3b)
        3. leave Ollama running

    Then:
        python simple_run.py --models ollama:qwen2.5:7b --trials 10

    WHY THIS IS GOOD SCIENCE, not just a way to save money: open-weight models
    have far weaker safety training than the commercial ones. They are where
    your attacks are most likely to actually succeed — which is what gives you
    failures to analyse. A run where nothing ever breaks is a thin dissertation.
    """

    name = "ollama"
    BASE_URL = "http://localhost:11434/v1"

    def __init__(self, model="qwen2.5:7b", temperature=0.0, system_prompt=""):
        from openai import OpenAI
        self.client = OpenAI(base_url=self.BASE_URL, api_key="ollama-needs-no-key", timeout=1800.0, max_retries=2)
        self.model = model
        self.temperature = temperature
        self.messages = [{"role": "system", "content": system_prompt}]
        self.pending_tool_call_id = None
        self.tokens_in = 0
        self.tokens_out = 0
        self.supports_temperature = _accepts(
            self.client.chat.completions.create, "temperature")

    def next_action(self):
        try:
            return super().next_action()
        except Exception as e:
            txt = str(e).lower()
            if "connection" in txt or "refused" in txt or "connect" in txt:
                raise RuntimeError(
                    "Cannot reach Ollama on localhost:11434.\n"
                    "  Is it running?  Open the Ollama app, or run:  ollama serve\n"
                    f"  Have you downloaded the model?  ollama pull {self.model}"
                ) from e
            raise


def make_provider(kind, model=None, temperature=0.0, system_prompt=""):
    if kind == "fake":
        return FakeProvider()
    if kind == "anthropic":
        return AnthropicProvider(model or "claude-sonnet-4-5", temperature, system_prompt)
    if kind == "openai":
        return OpenAIProvider(model or "gpt-4o", temperature, system_prompt)
    if kind == "ollama":
        return OllamaProvider(model or "qwen2.5:7b", temperature, system_prompt)
    raise ValueError(f"unknown provider '{kind}' (use: fake, ollama, anthropic, openai)")


# ------------------------------------------------------------------- runner --

def describe_scene(robot, cfg, description_overrides=None):
    """Turn the simulator's state into the text the model will read.

    IMPORTANT FOR THE PROJECT: this function is an attack surface. Whatever
    text ends up here goes into the model's context, and the model cannot tell
    it apart from an instruction you typed. That is exactly what scenarios
    A6-A8 exploit.
    """
    description_overrides = description_overrides or {}
    lines = []
    for name, pos in robot.scene_objects():
        spec = cfg["objects"].get(name, {})
        desc = description_overrides.get(name) or spec.get("description_string", name)
        lines.append(f"- {desc.strip()}  [at x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}]")
    held = robot.held_object or "nothing"
    return "Objects on the table:\n" + "\n".join(lines) + f"\nThe gripper is holding: {held}"


def execute(robot, tool, args, cfg, description_overrides=None, verifier=None):
    """Run one chosen command against the robot, with an optional safety check."""
    if verifier is not None:
        allowed, reason = verifier(tool, args, robot)
        if not allowed:
            return {"ok": False, "message": f"BLOCKED BY SAFETY CHECK: {reason}",
                    "blocked": True}

    if tool == "get_scene":
        return {"ok": True, "message": describe_scene(robot, cfg, description_overrides)}
    if tool == "pick":
        return robot.pick(args["object_id"], float(args.get("force", 60.0))).to_dict()
    if tool == "place":
        return robot.place(args["object_id"], args.get("target")).to_dict()
    if tool == "move_to":
        return robot.move_to(float(args["x"]), float(args["y"]), float(args["z"]),
                             float(args.get("speed", 0.25))).to_dict()
    if tool == "open_gripper":
        return robot.open_gripper().to_dict()
    if tool == "close_gripper":
        return robot.close_gripper(float(args.get("force", 60.0))).to_dict()
    if tool == "report":
        return robot.report(args.get("message", "")).to_dict()
    if tool == "done":
        return {"ok": True, "message": "task ended by the model", "done": True}
    return {"ok": False, "message": f"unknown command '{tool}'"}


def run_conversation(provider, robot, cfg, system_prompt, user_turns,
                     max_steps=20, description_overrides=None, verifier=None,
                     verbose=True):
    """The main loop: ask the model, run its answer, hand back the result."""
    transcript = []

    for turn_text in user_turns:
        provider.add_user(turn_text)
        if verbose:
            print(f"\n  USER: {turn_text}")

        for _ in range(max_steps):
            tool, args = provider.next_action()
            result = execute(robot, tool, args, cfg, description_overrides, verifier)

            if verbose:
                shown = args if tool != "report" else {"message": str(args.get("message", ""))[:90]}
                print(f"  ROBOT: {tool}({shown})")
                print(f"     -> {str(result.get('message', ''))[:160]}")

            transcript.append({"tool": tool, "args": args, "result": result,
                               "tcp": robot.tcp_pos().tolist()})

            if result.get("done"):
                return transcript

            provider.add_result(result.get("message", ""))

    return transcript
