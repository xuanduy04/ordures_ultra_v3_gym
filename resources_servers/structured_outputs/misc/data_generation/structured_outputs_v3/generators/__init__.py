
from generators.direct import generate_direct
from generators.error_correction import generate_error_correction
from generators.multistep import generate_multistep_related, generate_multistep_unrelated
from generators.schema_only import generate_schema_only
from generators.translation import generate_translation


ALL_GENERATORS = {
    "direct": generate_direct,
    "translation": generate_translation,
    "multistep_related": generate_multistep_related,
    "multistep_unrelated": generate_multistep_unrelated,
    "schema_only": generate_schema_only,
    "error_correction": generate_error_correction,
}
