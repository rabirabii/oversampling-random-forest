import random


class DynamicMessage:
    def __init__(self):
        self.success_messages = [
            "Based on our combined analysis, you appear to have a lower risk of diabetes. However, always consult with a healthcare professional for a proper evaluation.",
            "Our analysis suggests that you have a low probability of diabetes based on current health indicators. The fuzzy logic system confirms that your glucose, BMI, and age align with a lower risk profile. Nonetheless, it's always best to consult with a healthcare provider for personalized advice.",
            "According to our machine learning model, you appear to be at minimal risk for diabetes. While the model confidently indicates this, regular check-ups with your healthcare professional are essential to maintain good health.",
            "Both our machine learning and fuzzy logic models show that your risk of diabetes is low. Keep in mind that prevention is key, and maintaining healthy habits will help you stay on track. Feel free to discuss further with a healthcare expert.",
            "The results from our combined machine learning and fuzzy system suggest that your current risk for diabetes is below average. However, this is not a definitive diagnosis. Regular consultations and health monitoring are recommended.",
        ]

        self.warning_messages = [
            "Based on our combined analysis, you may have an elevated risk of diabetes. Please consult with a healthcare professional for a thorough evaluation and advice.",
            "The fuzzy logic system flags a higher risk of diabetes, particularly based on key factors like glucose and BMI. It's advisable to follow up with a healthcare professional for further tests and a more comprehensive evaluation.",
            "Our Random Forest model predicts a significant risk of developing diabetes based on your data. To ensure your well-being, we strongly recommend seeking professional medical advice for detailed guidance and potential next steps.",
            "Both the machine learning model and fuzzy logic assessment indicate a potentially elevated risk for diabetes. This suggests you may benefit from a more in-depth medical evaluation. Please consult with your doctor as soon as possible.",
            "The combined analysis of your data points to a moderate to high risk of diabetes. Early detection is key to prevention, so we urge you to schedule an appointment with a healthcare provider to explore appropriate actions.",
        ]

    def get_success_message(self, bmi, age, glucose):
        return random.choice(self.success_messages).format(
            bmi=bmi, age=age, glucose=glucose
        )

    def get_warning_message(self, bmi, age, glucose):
        return random.choice(self.warning_messages).format(
            bmi=bmi, age=age, glucose=glucose
        )
