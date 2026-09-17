# Methodology — worksheet for 3.1 and 3.2
# You write the answers here in rough notes. Full sentences not needed yet.
# I will then question the notes and help you turn them into your own prose.

## 3.1 Research design  (~300 words)

**Q1. The one thing you change.**
Between two otherwise identical runs, what is the single thing you switch?
(One sentence. This is your independent variable.)
>

**Q2. What stays fixed.**
When you switch it, what is held constant? List them — scenario, objects,
positions, system prompt, model, trial seed, anything else.
>

**Q3. What you measure.**
Two numbers come out of every run. What are they, and what is the difference
between them? (Hint: one describes the world, one describes the model's choice.
Say why you kept them separate instead of collapsing into one score.)
>

**Q4. Why repeat.**
Why is 1 run per condition not enough? Give the technical reason, not "to be safe".
>

**Q5. Why matched pairs.**
You could have compared a group of "checker on" runs against a different group of
"checker off" runs. Instead each scenario is its own pair. Why is that stronger?
>

**Q6. What you wrote down before running.**
scenarios.yaml has a `hypothesis` field per scenario, filled in before any run.
Why does that change what your results are worth?
>

**Q7. The unit.**
What is one data point? One action? One turn? One episode? Say it plainly.
>

---

## 3.2 System architecture  (~450 words + Figure 3.1)

**Q8. The chain.**
Name every box between "operator types a sentence" and "the world changes",
in order.
>

**Q9. The three inputs.**
Three different things feed the model. What are they? Which one is the attack
surface, and why is that the interesting one?
>

**Q10. The design argument — this is the important one.**
Why can a safety checker exist in your design but NOT inside an end-to-end
vision-language-action model? Answer in your own words. Do not write "because
it is modular" — say what a checker needs in order to be able to check anything.
>

**Q11. The arm.**
Your system prompts say UR5e. Your retained MuJoCo demo uses a Franka Panda.
An examiner will notice. Write the one sentence that explains this honestly.
>

**Q12. What the world actually is.**
In the experiment there is no physics engine. What does your `World` class
compute? Now write what it does NOT compute. Be specific — this is where you
earn "rigorous explanation of scope and limits".
>

**Q13. Why the menu is fixed.**
The model can only pick from 8 commands. It cannot invent a ninth. So where
does the danger come from instead?
>

---

## Notes to fold in later

- The `done` truncation defect (found 15 Sept) belongs in 3.6 and in the
  limitations. It was found by comparing run wall-clock times and action counts
  against expected conversation length. Frame it as method, not apology.
- The A10 / C4 scenarios were discarded because of it. Say so.
- Anthropic SDK v1 dropped `temperature`, so determinism is impossible.
  That is the real reason repeated trials are mandatory.
