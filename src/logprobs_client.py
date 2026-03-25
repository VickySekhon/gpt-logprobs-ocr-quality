"""
Converts a page excerpt to plain text file and caches it.
"""
import os
from pathlib import Path
from scan2latex_entropy import encode_image, chat, EXCLUDE_TOKENS

def transcribe_with_logprobs(image_path):
    encoded_image = encode_image(image_path)

    # system_prompt = "You are an OCR engine for historical English newspapers. You will transcribe images of historical English newspapers into plain text **without altering** any information present in the images. You will only output the transcribed plain text and nothing else."

    # user_text = "Transcribe this image of a historical English newspaper excerpt. Preserve capitalization, punctuation, whitespaces, and special characters such as currency symbols, fractions, section dividers, quotations, dashes, and apostrophes as seen in the provided image. Do not worry about indentation and do not use any code fences. The output should be plain text."

    system_prompt = """ 
    You are an OCR engine for historical English newspapers.
    Your task is to transcribe the provided image exactly as it appears.
    Do not:
    - Correct spelling
    - Expand abbreviations
    - Normalize ligatures
    - Correct punctuation
    - Modernize language
    - Interpret or infer missing content
    Preserve the original text exactly, except where explicitly instructed otherwise.
    Output only the transcribed plain text and nothing else.
    """

    user_text = """
    Transcribe this image of a historical English newspaper excerpt.
    Preserve exactly as shown:
    - Original capitalization
    - Original spelling (including archaic spellings)
    - Original punctuation
    - Special characters (currency symbols, fractions, quotation marks, dashes, apostrophes, etc.)
    - Section dividers exactly as they appear
    Line-break handling rules:
    - If a word is split across lines using a hyphen at the end of a line, remove the hyphen and reconstruct the full word on a single line.
    - Do NOT remove hyphens that are part of compound words within a line.
    If text is unclear, output your best guess based only on visible characters. **Do not** mark uncertainty and **do not** insert placeholders.
    Return only the plain text transcription. Do not add commentary or formatting.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]
    resp = chat(messages)
    choice = resp.choices[0]

    transcript_text = choice.message.content.strip()
    token_logprobs = [
        t
        for t in choice.logprobs.content
        if t.token not in EXCLUDE_TOKENS and t.token.strip()
    ]
    
    return transcript_text, token_logprobs


if __name__ == "__main__":
    IMAGE_PATH = Path(os.path.join(os.getcwd(), "data/images/3200797037.jpg"))
    transcribe_with_logprobs(IMAGE_PATH)
    