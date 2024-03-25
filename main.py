import streamlit as st
from models.carousel_generator import generate_carousel

def main():
    st.title('StreamIt App')
    
    # Input box for user input
    user_input = st.text_input('Enter your question :')
    
    if st.button('Process'):
        if user_input:
            try:
                query = str(user_input)
                result = generate_carousel(query)
                st.success(f'Result: {result}')
            except ValueError:
                st.error('Please enter a valid number.')
        else:
            st.warning('Please enter a number above.')

if __name__ == '__main__':
    main()
