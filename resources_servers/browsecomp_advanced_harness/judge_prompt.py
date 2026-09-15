

# from: https://github.com/openai/simple-evals/blob/5e623c2b400af62a1278e23595f95b0853d7fe8a/browsecomp_eval.py#L26-L45 and https://github.com/openai/simple-evals/blob/5e623c2b400af62a1278e23595f95b0853d7fe8a/simpleqa_eval.py#L12-L104. We keep the output format of browsecomp.

JUDGE_PROMPT_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not \
based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. \
Put the extracted answer as 'None' if there is no exact, final answer to \
extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect \
based on [correct_answer], focusing only on if there are meaningful \
differences between [correct_answer] and the extracted_final_answer. Do not \
comment on any background to the problem, do not attempt to solve the \
problem, do not argue for any answer different than [correct_answer], focus \
only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] \
given above, or is within a small margin of error for numerical problems. \
Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, \
non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from \
[response]. Put 100 if there is no confidence score available."""
