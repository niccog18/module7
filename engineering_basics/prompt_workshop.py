import os


# --- Mock response function from the Guided Example ---
def mock_response(prompt):
    """Simulates how prompt quality affects response quality."""
    prompt_lower = prompt.lower()

    # Detect if the prompt has framing
    has_role = any(
        word in prompt_lower
        for word in ["you are", "act as", "your role"]
    )

    # Detect if the prompt has constraints
    has_constraints = any(
        word in prompt_lower
        for word in [
            "under 100",
            "under 150",
            "in 3 sentences",
            "as a table",
            "as json",
            "format",
            "bullet",
        ]
    )

    # Detect if the prompt has examples
    has_examples = (
        "input:" in prompt_lower
        or "example:" in prompt_lower
    )

    quality_score = sum([
        has_role,
        has_constraints,
        has_examples
    ])

    if quality_score == 0:
        return (
            "[MOCK — Vague prompt detected]\n"
            "This is a generic response because the prompt "
            "did not provide much direction."
        )

    elif quality_score == 1:
        return (
            "[MOCK — Decent prompt]\n"
            "This response is somewhat focused because the prompt "
            "provided one useful instruction."
        )

    else:
        return (
            "[MOCK — Well-engineered prompt]\n"
            "This response is focused and structured because the "
            "prompt provided framing, constraints, and/or examples."
        )


# --- Try real API, fall back to mock ---
def get_response(prompt):
    """Try real API, fall back to mock."""
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        import openai

        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=300
        )

        return response.choices[0].message.content

    else:
        return mock_response(prompt)


# ============================================================
# TASK 1 — CODE EXPLANATION
# ============================================================

print("=" * 70)
print("TASK 1: CODE EXPLANATION")
print("=" * 70)

bad_prompt_1 = """
What does st.session_state do in Streamlit?
"""

good_prompt_1 = """
You are a patient Python and Streamlit instructor.

Explain what st.session_state does in Streamlit to a beginner.

Requirements:
- Explain its purpose in simple language.
- Give one practical example.
- Keep the explanation under 100 words.
- Avoid unnecessary technical jargon.
"""

print("\n--- BAD PROMPT ---")
print(bad_prompt_1)

print("--- BAD OUTPUT ---")
print(get_response(bad_prompt_1))

print("\n--- GOOD PROMPT ---")
print(good_prompt_1)

print("--- GOOD OUTPUT ---")
print(get_response(good_prompt_1))


# ============================================================
# TASK 2 — DATA FORMATTING
# ============================================================

print("\n" + "=" * 70)
print("TASK 2: DATA FORMATTING")
print("=" * 70)

bad_prompt_2 = """
Turn these tasks into JSON:

Finish Python homework, high priority, in progress.
Study for quiz, medium priority, not started.
Submit project, high priority, not started.
"""

good_prompt_2 = """
You are a data-formatting assistant.

Convert the following natural-language task list into a JSON array.

Each object must contain exactly these three fields:
- title
- priority
- status

Use these allowed priority values:
high, medium, low

Use these allowed status values:
not started, in progress, completed

Return ONLY valid JSON.
Do not include explanations or markdown.

Example:

Input:
"Finish homework, high priority, in progress"

Output:
{"title": "Finish homework", "priority": "high", "status": "in progress"}

Now convert:

Input:
"Finish Python homework, high priority, in progress.
Study for quiz, medium priority, not started.
Submit project, high priority, not started."
"""

print("\n--- BAD PROMPT ---")
print(bad_prompt_2)

print("--- BAD OUTPUT ---")
print(get_response(bad_prompt_2))

print("\n--- GOOD PROMPT ---")
print(good_prompt_2)

print("--- GOOD OUTPUT ---")
print(get_response(good_prompt_2))


# ============================================================
# TASK 3 — SYSTEM PROMPT DESIGN
# ============================================================

print("\n" + "=" * 70)
print("TASK 3: SYSTEM PROMPT DESIGN")
print("=" * 70)

bad_prompt_3 = """
Answer questions about my course notes.
"""

good_prompt_3 = """
You are a Course Study Assistant helping a student understand
their course material.

Your role:
Answer student questions using ONLY the provided course-note
context.

Rules:
1. Do not use outside knowledge or make up information.
2. If the answer cannot be found in the provided context, say:
   "I don't know based on the provided course notes."
3. Keep every answer under 150 words.
4. Include the source document name for the information used.
5. Be clear and concise.

Format your response as:

Answer:
<answer>

Source:
<source document name>

Example:

Context:
"Streamlit's st.session_state allows values to persist between
script reruns."

Question:
"What does st.session_state do?"

Answer:
"st.session_state allows values to persist between Streamlit
script reruns."

Source:
"streamlit_notes.md"
"""

print("\n--- BAD PROMPT ---")
print(bad_prompt_3)

print("--- BAD OUTPUT ---")
print(get_response(bad_prompt_3))

print("\n--- GOOD PROMPT ---")
print(good_prompt_3)

print("--- GOOD OUTPUT ---")
print(get_response(good_prompt_3))


# ============================================================
# FINAL SYSTEM PROMPT
# ============================================================

print("\n" + "=" * 70)
print("TASK 3 — FINAL COURSE STUDY ASSISTANT SYSTEM PROMPT")
print("=" * 70)

print(good_prompt_3)


# ============================================================
# KEY TAKEAWAY
# ============================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAY")
print("=" * 70)

print(
    "Good prompts use framing, specificity, constraints, "
    "and/or few-shot examples to produce more focused results."
)