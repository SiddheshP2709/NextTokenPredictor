"""Unit tests for text and code tokenization pipelines."""

import pytest
from next_token_predictor.tokenization.text_tokenizer import TextTokenizer
from next_token_predictor.tokenization.code_tokenizer import CodeTokenizer


def test_text_tokenizer_from_text():
    sample = "The prince was silent. The count walked into the room."
    tokenizer = TextTokenizer.build_from_text(sample)

    assert "." in tokenizer.stoi
    assert "prince" in tokenizer.stoi
    assert len(tokenizer) > 5

    tokens = tokenizer.encode("The prince walked.")
    assert len(tokens) == 4
    decoded = tokenizer.decode(tokens)
    assert "prince" in decoded


def test_code_tokenizer_lexing():
    tokenizer = CodeTokenizer()
    cpp_snippet = "int calculate_sum(int a, int b) { return a + b; }"
    tokens = tokenizer.tokenize_code(cpp_snippet)

    expected_tokens = [
        "int", "calculate_sum", "(", "int", "a", ",", "int", "b", ")",
        "{", "return", "a", "+", "b", ";", "}"
    ]
    assert tokens == expected_tokens


def test_code_tokenizer_special_tokens():
    tokens_list = ["int", "NULL", "nullptr", "def", "None", "{", "}", ";"]
    stoi = {t: i for i, t in enumerate(tokens_list)}
    tokenizer = CodeTokenizer(stoi=stoi)

    encoded = tokenizer.encode("int x = NULL;")
    assert len(encoded) > 0
    assert tokenizer.stoi["NULL"] in encoded


def test_code_tokenizer_from_csv(tmp_path):
    csv_file = tmp_path / "vocab.csv"
    csv_file.write_text("token,freq\nint,100\ncalculate_sum,50\nNULL,20\nreturn,10\n", encoding="utf-8")

    tokenizer = CodeTokenizer.from_csv(str(csv_file))
    assert len(tokenizer) == 4
    assert tokenizer.stoi["int"] == 0
    assert tokenizer.itos[0] == "int"
    assert tokenizer.itos[1] == "calculate_sum"
    assert tokenizer.itos[2] == "NULL"
    assert tokenizer.itos[3] == "return"
    # Ensure itos and stoi are inverses
    for token, idx in tokenizer.stoi.items():
        assert tokenizer.itos[idx] == token

