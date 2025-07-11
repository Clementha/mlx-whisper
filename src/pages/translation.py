import streamlit as st
from translate import Translator

def translate_to_welsh(text):
    translator = Translator(to_lang="cy")
    try:
        return translator.translate(text)
    except Exception as e:
        return f"Translation error: {e}"

def main():
    st.title("English to Welsh Translation")

    english_input = st.text_area("Enter English Text", height=100)

    if st.button("Translate to Welsh"):
        if english_input.strip():
            welsh_translation = translate_to_welsh(english_input.strip())
            st.success("Welsh Translation:")
            st.text_area("Welsh", welsh_translation, height=100)
        else:
            st.warning("Please enter some English text.")

if __name__ == "__main__":
    main()
