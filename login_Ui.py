import streamlit as st

from auth import (
    signup_user,
    login_user,
    retrieve_user_threads,
)


def show_login_page():

    st.title("🔐 Multi Utility Chatbot")

    st.markdown(
        "Login or create an account to continue."
    )

    login_tab, signup_tab = st.tabs(
        [
            "Login",
            "Sign Up",
        ]
    )

    # -------------------
    # Login
    # -------------------
    with login_tab:

        username = st.text_input(
            "Username",
            key="login_username",
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
        )

        if st.button(
            "Login",
            use_container_width=True,
        ):

            user_id = login_user(
                username,
                password,
            )

            if user_id:

                st.session_state[
                    "logged_in"
                ] = True

                st.session_state[
                    "user_id"
                ] = user_id

                st.session_state[
                    "username"
                ] = username

                st.session_state[
                    "chat_threads"
                ] = retrieve_user_threads(
                    user_id
                )

                st.success(
                    "Login successful"
                )
                

                st.rerun()

            else:

                st.error(
                    "Invalid username or password"
                )

    # -------------------
    # Signup
    # -------------------
    with signup_tab:

        username = st.text_input(
            "Username",
            key="signup_username",
        )

        password = st.text_input(
            "Password",
            type="password",
            key="signup_password",
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            key="signup_confirm_password",
        )

        if st.button(
            "Create Account",
            use_container_width=True,
        ):

            if not username.strip():

                st.error(
                    "Username is required"
                )

            elif len(password) < 6:

                st.error(
                    "Password must be at least 6 characters"
                )

            elif password != confirm_password:

                st.error(
                    "Passwords do not match"
                )

            else:

                success, message = signup_user(
                    username,
                    password,
                )

                if success:

                    st.success(
                        "Account created successfully. Please login."
                    )

                else:

                    st.error(
                        message
                    )