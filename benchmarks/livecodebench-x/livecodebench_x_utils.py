
"""Multilingual LiveCodeBench-X constants — mirrors NeMo Skills' versions in
`nemo_skills/dataset/livecodebench-x/livecodebench_x_utils.py`."""

SUPPORTED_LANGUAGES = ["de", "es", "fr", "ja"]
SUPPORTED_VERSIONS = ["v5", "v6"]

# English instruction (matches Skills' EN_INSTRUCTION).
EN_INSTRUCTION = "Here is a problem for which you need to generate executable code in the Python programming language."

# Language-specific instructions copied verbatim from Skills'
# `livecodebench_x_utils.CODEGEN_INSTRUCTIONS`. Translations match Skills'
# `eval/livecodebench/python_codegen_{lang}.yaml`.
CODEGEN_INSTRUCTIONS = {
    "de": ("Hier ist ein Problem, für das Sie ausführbaren Code in der Programmiersprache Python generieren müssen."),
    "es": (
        "Aquí tienes un problema para el cual necesitas generar un código "
        "ejecutable en el lenguaje de programación Python."
    ),
    "fr": ("Voici un problème pour lequel vous devez générer du code exécutable en langage de programmation Python."),
    "ja": ("以下は、Pythonプログラミング言語で実行可能なコードを生成する必要がある問題です。"),
}
