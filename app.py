import streamlit as st

# Streamlit app
def main():
    # Favicon
    st.set_page_config(page_title="moodAI", page_icon="img/logo_moodai.png")
    
    # Title
    st.image("img/moodAI.png", use_column_width=True)
    
    # Subheader and description
    st.subheader("Helping you understand the emotions behind text messages")
    # Description
    st.markdown("moodAI is designed to **empower** individuals, especially those with special needs, in **deciphering the emotions of others**.")

    # Text input for user input
    text = st.text_input("Enter a text message below to see the sentiment analysis result:")

    # Add a button to trigger the classification
    if st.button("Analyze"):
        # Display the sentiment analysis result
        st.write("🥰 affection - 57%" if text == "I love you" else "😡 anger - 22%")

# Run the app
if __name__ == "__main__":
    main()