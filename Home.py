import streamlit as st

st.set_page_config(
    page_title="Yoga Pose Classification",
    page_icon="🧘‍♀️",
    layout="wide"
)

def main():
    st.write("# Welcome👋")
    st.markdown("""
        ## About Us:
        We're building an AI-powered Yoga app with real-time pose detection and personalized recommendations. 
        Help us to revolutionize wellness with Machine Learning. 

        ## Choose a page from the sidebar to get started:
        - **User**: Practice yoga poses with real-time feedback
        - **Developer**: Access developer tools and settings (Requires Authentication)
    """)

if __name__ == "__main__":
    main()