from groq import Groq
import os
import dotenv
import time
dotenv.load_dotenv()

KEY = os.getenv("GROQ_API_KEY")

prompt = """
        Analyze the weather data provided with the following parameters:
        - Temperature (°C) at 2 meters above ground
        - Precipitation (mm) during the last hour
        - Wind speed (m/s) at 10 meters

        Focus on identifying trends and patterns that could affect weather conditions over the specified period. Based on the historical data and current values, predict the likelihood of rainfall and potential weather changes. Provide a detailed explanation of the prediction, highlighting the most influential factors and their impact on the result.

        The output should include:
        1. A concise summary of weather trends.
        2. A probability assessment for rainfall.
        3. Recommendations for the user based on predicted weather conditions.
        4. A visual representation of the data (table) showing temperature, precipitation, and wind speed trends over time.
        """

# Initialize the Groq client with your API key
client = Groq(
    api_key=KEY  # replace with your API key
)


def analyze_weather_data(data):
    try:
        # Call the Groq AI model to generate a response based on the prompt
        responder = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"This is the data:{data}, {prompt}"},
            ],
            model="mixtral-8x7b-32768",
            temperature=0.5,
        )

        # Extract the AI-generated message from the response
        message = responder.choices[0].message.content
        return message

    except Exception as e:
        if "rate limit" in str(e):
            wait_time = 20  # Wait a few seconds before retrying
            # print(f"Rate limit hit. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            return analyze_weather_data(data)  # Retry the request
        else:
            return f"An error occurred during the AI analysis: {e}. Please check your internet connection or API key."

