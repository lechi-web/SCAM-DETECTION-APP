import os
import pickle
import re

import streamlit as st
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# SCAMBUSTER AI
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ScamBuster AI",
    page_icon="🛡️",
    layout="centered"
)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

except FileNotFoundError:
    st.error(
        "❌ model.pkl was not found. "
        "Please train your model first."
    )
    st.stop()

except Exception as error:
    st.error(
        f"❌ Could not load the machine learning model: {error}"
    )
    st.stop()


# ============================================================
# LOAD GROQ API KEY
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "❌ GROQ_API_KEY was not found. "
        "Please check your .env file."
    )
    st.stop()


try:
    client = Groq(api_key=api_key)

except Exception as error:
    st.error(
        f"❌ Could not connect to Groq: {error}"
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ ScamBuster AI")

st.caption(
    "AI-powered scam detection assistant that analyzes suspicious "
    "messages, identifies scam patterns, and explains potential threats."
)

st.divider()


# ============================================================
# MESSAGE INPUT
# ============================================================

st.subheader("📩 Paste a Suspicious Message")

input_msg = st.text_area(
    "Message",
    height=180,
    placeholder=(
        "Example:\n\n"
        "Your account has been suspended. "
        "Click here immediately to verify..."
    ),
    label_visibility="collapsed"
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Message",
    use_container_width=True
):

    if not input_msg.strip():

        st.warning(
            "⚠️ Please enter a message first."
        )

    else:

        status = st.empty()

        status.info(
            "🤖 ScamBuster AI is analyzing your message..."
        )

        try:

            # ====================================================
            # NORMALIZE MESSAGE
            # ====================================================

            message = input_msg.strip()
            message_lower = message.lower()


            # ====================================================
            # MACHINE LEARNING PREDICTION
            # ====================================================

            prediction = model.predict([message])[0]

            scam_probability = None

            try:

                probabilities = model.predict_proba([message])[0]

                classes = list(model.classes_)

                if 1 in classes:

                    scam_index = classes.index(1)

                    scam_probability = float(
                        probabilities[scam_index]
                    )

            except Exception:

                scam_probability = None


            # ====================================================
            # THREAT INDICATOR: URL / LINK
            # ====================================================

            url_found = bool(
                re.search(
                    r"(https?://|www\.|bit\.ly|tinyurl|t\.co/)",
                    message_lower
                )
            )

            link_request_words = [
                "click here",
                "click the link",
                "click this link",
                "click on the link",
                "visit this link",
                "open the link",
                "follow this link",
                "tap here",
                "verify here",
                "use this link"
            ]

            link_request_found = any(
                word in message_lower
                for word in link_request_words
            )

            suspicious_link = (
                url_found or link_request_found
            )


            # ====================================================
            # THREAT INDICATOR: URGENCY
            # ====================================================

            urgency_words = [
                "urgent",
                "immediately",
                "act now",
                "do this now",
                "right now",
                "as soon as possible",
                "within 24 hours",
                "within 48 hours",
                "last warning",
                "expires",
                "expire",
                "suspended",
                "suspension",
                "blocked",
                "lock your account",
                "account will be closed",
                "avoid closure",
                "avoid suspension",
                "limited time"
            ]

            urgency_found = any(
                word in message_lower
                for word in urgency_words
            )


            # ====================================================
            # THREAT INDICATOR: FINANCIAL KEYWORDS
            # ====================================================

            financial_words = [
                "₦",
                "naira",
                "money",
                "payment",
                "transfer",
                "bank",
                "banking",
                "debit",
                "credit",
                "loan",
                "cash",
                "reward",
                "grant",
                "prize",
                "investment",
                "deposit",
                "withdraw",
                "transaction",
                "wallet",
                "pos"
            ]

            financial_found = any(
                word in message_lower
                for word in financial_words
            )


            # ====================================================
            # THREAT INDICATOR: OTP
            # ====================================================

            otp_found = bool(
                re.search(
                    r"\botp\b|one[- ]time password",
                    message_lower
                )
            )


            # ====================================================
            # THREAT INDICATOR: SENSITIVE INFORMATION
            # ====================================================

            sensitive_words = [
                "password",
                "passcode",
                "pin",
                "bvn",
                "nin",
                "cvv",
                "card number",
                "account number",
                "login details",
                "login credentials",
                "verification code",
                "security code",
                "personal information",
                "identity information",
                "kyc",
                "update your details",
                "update your information",
                "verify your identity",
                "confirm your identity"
            ]

            sensitive_found = any(
                word in message_lower
                for word in sensitive_words
            )


            # ====================================================
            # GROQ AI ANALYSIS
            # ====================================================

            analysis_prompt = f"""
You are ScamBuster AI, a cybersecurity education assistant.

Analyze ONLY the message provided below.

MESSAGE:
{message}

IMPORTANT RULES:

1. Only report evidence that actually appears in the message.

2. NEVER invent a red flag.

3. NEVER say the message contains a spelling error unless
   an actual spelling error is clearly present.

4. NEVER claim that a suspicious URL exists unless an actual
   URL or link request appears in the message.

5. NEVER claim urgency unless the message contains actual
   urgent, threatening, time-pressure, suspension, blocking,
   or immediate-action language.

6. Do not call a message a scam simply because it mentions
   a bank, money, an account, payment, or an appointment.

7. A normal appointment confirmation can be legitimate.

8. Requests for OTPs, passwords, PINs, BVN, NIN, CVV,
   account numbers, or other sensitive information are
   strong warning signs.

9. Suspicious links or requests to click links are strong
   warning signs.

10. Threats, account suspension, blocking, expiration,
    or pressure to act immediately are warning signs.

11. If there are no meaningful scam indicators, classify
    the message as LIKELY LEGITIMATE.

12. Do not assume information that is not present.

13. Keep the explanation beginner-friendly.

14. Do not use markdown headings.

15. Do not use emojis.

16. Use exactly one verdict:
    SCAM
    SUSPICIOUS
    LIKELY LEGITIMATE

Return EXACTLY this structure:

VERDICT: [SCAM / SUSPICIOUS / LIKELY LEGITIMATE]

CATEGORY: [Phishing / Banking Scam / Job Scam / Investment Scam / Delivery Scam / Other / None]

RED_FLAGS:
- [Only evidence-based red flag]
- [Only evidence-based red flag]
- [Only evidence-based red flag]

EXPLANATION:
[2-4 sentences explaining the result and teaching the user something useful.]

RECOMMENDATION:
[One short paragraph explaining what the user should do.]
"""


            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                temperature=0.1
            )

            analysis = (
                response.choices[0]
                .message
                .content
                .strip()
            )


            status.empty()


            # ====================================================
            # EXTRACT AI VERDICT
            # ====================================================

            analysis_upper = analysis.upper()

            if "VERDICT: SCAM" in analysis_upper:

                ai_verdict = "SCAM"

            elif "VERDICT: SUSPICIOUS" in analysis_upper:

                ai_verdict = "SUSPICIOUS"

            elif "VERDICT: LIKELY LEGITIMATE" in analysis_upper:

                ai_verdict = "LIKELY LEGITIMATE"

            else:

                ai_verdict = "SUSPICIOUS"


            # ====================================================
            # EXTRACT CATEGORY
            # ====================================================

            category_match = re.search(
                r"CATEGORY:\s*(.+)",
                analysis,
                re.IGNORECASE
            )

            if category_match:

                category = (
                    category_match
                    .group(1)
                    .strip()
                )

            else:

                category = "None"


            category = category.replace("*", "").strip()


            # ====================================================
            # EXTRACT RED FLAGS
            # ====================================================

            red_flags = []

            red_flags_match = re.search(
                r"RED_FLAGS:\s*(.*?)(?=\n\s*EXPLANATION:|\Z)",
                analysis,
                re.IGNORECASE | re.DOTALL
            )

            if red_flags_match:

                red_flags_text = (
                    red_flags_match
                    .group(1)
                    .strip()
                )

                for line in red_flags_text.splitlines():

                    line = line.strip()

                    if line.startswith("-"):

                        line = line[1:].strip()

                    if line:

                        red_flags.append(line)


            if not red_flags:

                red_flags = [
                    "No strong scam indicators detected."
                ]


            # ====================================================
            # EXTRACT EXPLANATION
            # ====================================================

            explanation_match = re.search(
                r"EXPLANATION:\s*(.*?)(?=\n\s*RECOMMENDATION:|\Z)",
                analysis,
                re.IGNORECASE | re.DOTALL
            )

            if explanation_match:

                explanation = (
                    explanation_match
                    .group(1)
                    .strip()
                )

            else:

                explanation = (
                    "Review the message carefully and verify "
                    "unexpected requests independently."
                )


            # ====================================================
            # EXTRACT RECOMMENDATION
            # ====================================================

            recommendation_match = re.search(
                r"RECOMMENDATION:\s*(.*)",
                analysis,
                re.IGNORECASE | re.DOTALL
            )

            if recommendation_match:

                recommendation = (
                    recommendation_match
                    .group(1)
                    .strip()
                )

            else:

                recommendation = (
                    "Verify the message independently before "
                    "taking any action."
                )


            # ====================================================
            # DETERMINE FINAL RESULT
            # ====================================================

            strong_indicators = (
                suspicious_link
                or otp_found
                or sensitive_found
            )


            # Strong technical indicators override an
            # incorrect "legitimate" AI response.

            if strong_indicators:

                final_result = "SCAM"
                risk_level = "HIGH"


            elif ai_verdict == "SCAM":

                # If AI says scam but there are no strong
                # technical indicators, treat it as suspicious
                # instead of automatically calling it a scam.

                if urgency_found:

                    final_result = "SUSPICIOUS"
                    risk_level = "MEDIUM"

                elif prediction == 1:

                    final_result = "SUSPICIOUS"
                    risk_level = "MEDIUM"

                else:

                    final_result = "SUSPICIOUS"
                    risk_level = "MEDIUM"


            elif ai_verdict == "SUSPICIOUS":

                final_result = "SUSPICIOUS"
                risk_level = "MEDIUM"


            elif prediction == 1:

                if scam_probability is not None:

                    if scam_probability >= 0.70:

                        final_result = "SCAM"
                        risk_level = "HIGH"

                    elif scam_probability >= 0.45:

                        final_result = "SUSPICIOUS"
                        risk_level = "MEDIUM"

                    else:

                        final_result = "LIKELY LEGITIMATE"
                        risk_level = "LOW"

                else:

                    final_result = "SUSPICIOUS"
                    risk_level = "MEDIUM"


            else:

                final_result = "LIKELY LEGITIMATE"
                risk_level = "LOW"


            # ====================================================
            # CATEGORY CORRECTION
            # ====================================================

            if final_result == "LIKELY LEGITIMATE":

                category = "None"

            elif category.lower() in [
                "none",
                "unknown",
                "n/a",
                "not applicable"
            ]:

                if suspicious_link:

                    category = "Phishing"

                elif otp_found or sensitive_found:

                    category = "Phishing"

                elif financial_found:

                    category = "Financial Scam"

                else:

                    category = "Other"


            # ====================================================
            # SUCCESS MESSAGE
            # ====================================================

            st.success(
                "✅ Scan completed successfully!"
            )

            st.divider()


            # ====================================================
            # SECURITY REPORT
            # ====================================================

            st.header(
                "🛡️ ScamBuster AI Security Report"
            )


            # ====================================================
            # RISK LEVEL
            # ====================================================

            if final_result == "SCAM":

                st.error(
                    "🔴 HIGH RISK — LIKELY SCAM"
                )

                st.error(
                    "🚨 ScamBuster Result: SCAM DETECTED"
                )


            elif final_result == "SUSPICIOUS":

                st.warning(
                    "🟠 MEDIUM RISK — SUSPICIOUS"
                )

                st.warning(
                    "⚠️ ScamBuster Result: SUSPICIOUS MESSAGE"
                )


            else:

                st.success(
                    "🟢 LOW RISK — NO STRONG SCAM INDICATORS"
                )

                st.success(
                    "✅ ScamBuster Result: LIKELY LEGITIMATE"
                )


            # ====================================================
            # CATEGORY
            # ====================================================

            st.write(
                f"🏷️ **Scam Category:** {category}"
            )


            # ====================================================
            # THREAT INDICATORS
            # ====================================================

            st.subheader(
                "⚠️ Threat Indicators"
            )

            st.write(
                "🌐 **Suspicious URL or Link Request:** "
                + (
                    "✅ Yes"
                    if suspicious_link
                    else "❌ No"
                )
            )

            st.write(
                "⚡ **Urgent Language:** "
                + (
                    "✅ Yes"
                    if urgency_found
                    else "❌ No"
                )
            )

            st.write(
                "💰 **Financial Keywords:** "
                + (
                    "✅ Yes"
                    if financial_found
                    else "❌ No"
                )
            )

            st.write(
                "🔐 **OTP Mentioned:** "
                + (
                    "✅ Yes"
                    if otp_found
                    else "❌ No"
                )
            )

            st.write(
                "🔑 **Sensitive Account Information:** "
                + (
                    "✅ Yes"
                    if sensitive_found
                    else "❌ No"
                )
            )


            # ====================================================
            # WHY FLAGGED
            # ====================================================

            st.subheader(
                "🚩 Why Was This Message Flagged?"
            )

            if final_result == "LIKELY LEGITIMATE":

                st.write(
                    "No strong scam indicators detected."
                )

            else:

                for flag in red_flags:

                    st.write(
                        f"🔴 {flag}"
                    )


            # ====================================================
            # SCAMBUSTER EXPLAINS
            # ====================================================

            st.subheader(
                "🤖 ScamBuster Explains"
            )

            st.write(
                explanation
            )


            # ====================================================
            # RECOMMENDED ACTION
            # ====================================================

            st.subheader(
                "🛡️ Recommended Action"
            )

            st.write(
                recommendation
            )


            # ====================================================
            # DETECTION DETAILS
            # ====================================================

            st.divider()

            with st.expander(
                "🔍 View Detection Details"
            ):

                st.subheader(
                    "Machine Learning Result"
                )

                if prediction == 1:

                    st.error(
                        "🚨 ML Model: SCAM"
                    )

                else:

                    st.info(
                        "ℹ️ ML Model: LIKELY LEGITIMATE"
                    )


                st.subheader(
                    "AI Result"
                )

                if ai_verdict == "SCAM":

                    st.error(
                        "🚨 AI: SCAM"
                    )

                elif ai_verdict == "SUSPICIOUS":

                    st.warning(
                        "⚠️ AI: SUSPICIOUS"
                    )

                else:

                    st.success(
                        "✅ AI: LIKELY LEGITIMATE"
                    )


                st.subheader(
                    "Final ScamBuster Result"
                )

                if final_result == "SCAM":

                    st.error(
                        "🔴 SCAM"
                    )

                elif final_result == "SUSPICIOUS":

                    st.warning(
                        "🟠 SUSPICIOUS"
                    )

                else:

                    st.success(
                        "🟢 LIKELY LEGITIMATE"
                    )


                # ====================================================
                # ML CONFIDENCE
                # ====================================================

                if scam_probability is not None:

                    if prediction == 1:

                        st.write(
                            f"🧠 ML Model Confidence: "
                            f"{scam_probability:.2%} scam likelihood"
                        )

                    else:

                        legitimate_probability = (
                            1 - scam_probability
                        )

                        st.write(
                            f"🧠 ML Model Confidence: "
                            f"{legitimate_probability:.2%} likely legitimate"
                        )


        except Exception as error:

            status.empty()

            st.error(
                "❌ Something went wrong while analyzing "
                f"the message: {error}"
            )