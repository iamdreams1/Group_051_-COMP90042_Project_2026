# 4. Report Writing

> ⚠️ **You MUST use the [ACL template](https://2023.aclweb.org/calls/style_and_formatting/) when writing your report.**

You must use **LATEX** for writing your report. You must include your group number under the title (using the `\author` field in LATEX).

Make sure to change the template setting from `\usepackage[review]{acl}` to `\usepackage[final]{acl}` to generate the final version. We will not accept reports that are longer than the stated limits below, or otherwise violate the style requirements.

The report should be submitted as a **PDF** and contain **no more than seven (7) pages of content**, excluding team contribution and references. **An appendix is NOT allowed.** Therefore, you should carefully consider the information you want to include in the report to build a coherent and concise narrative.

---

## Suggested Report Structure

### Title
The title of your project and Group Name.

### Abstract
An abstract should concisely (**less than 300 words**) motivate the problem, describe your aims, describe your contribution, and highlight your main finding(s).

### Introduction
The introduction explains the problem, why it's difficult, interesting, or important, how and why current methods succeed/fail at the problem, and explains the key ideas of your approach and results. Though an introduction covers similar material as an abstract, the introduction gives more space for motivation, detail, and references to existing work and captures the reader's interest.

### Approach
This section details your approach(es) to the problem. For example, this is where you describe the architecture of your neural network(s), and any other key methods or algorithms.

- You should be specific when describing your main approaches — you probably want to include equations and figures.
- You should also describe your baseline(s). Depending on space constraints, and how standard your baseline is, you might do this in detail, or simply refer the reader to some other paper for the details.
- If any part of your approach is original, make it clear (so we can give you credit!). For models and techniques that aren't yours, provide references.
- As you're setting up equations, notation, and the like, be sure to agree on a fixed technical vocabulary (that you've defined, or is well-defined in the literature) before writing and use it consistently throughout the report! This will make it easier for the teaching team to follow and is nice practice for research writing in general.

### Experiments
This section contains the following.

- **Evaluation method**: If you're defining your own metrics (for diagnostic purposes), be clear as to what you're hoping to measure with each evaluation method (whether quantitative or qualitative, automatic or human-defined!), and how it's defined.
- **Experimental details**: Report how you ran your experiments (e.g., model configurations, learning rate, training time, etc.)

### Results
Report the quantitative results that you have found so far. Use a table or plot to compare results and compare against baselines. You **must report dev results**, and also **test results if you participate in the leaderboard**. When analysing your results, consider the following:

- Are they what you expected?
- Better than you expected?
- Is it worse than you expected?
- Why do you think that is?
- What does that tell you about your approach?

### Conclusion
Summarise the main findings of your project, and what you have learnt. Highlight your achievements, and note the primary limitations of your work. If you like, you can describe avenues for future work.

### Team contributions
*(doesn't count towards the page limit)*

If you are a multi-person team, briefly describe the contributions of each member of the team.

### References
*(doesn't count towards the page limit)*

Your references section should be produced using **BibTeX**.

---

## ⚠️ Important

The report must be a **faithful description of the system you implemented**.
If there is a mismatch between the report and the submitted code, marks will be deducted.

> A strong report not only presents results, but also provides clear insight into **_why_** the method works (or fails).
