# Data Analyst Agent

A chat-style Streamlit application that creates data visualizations, interprets
approved charts with Gemini, and synthesizes approved reports into a final
summary.

## Setup

1. Install the dependencies with `pip install -r requirements.txt`.
2. Set `GOOGLE_API_KEY` (or `GOOGLE_GENAI_API_KEY`) in the environment or in a
   local `.env` file. Do not commit credentials.
3. Start the application with `streamlit run main.py`.

The configured model is `gemini-3.1-flash-lite`.

## Using the application

1. Upload a clean `.csv` dataset. The application stores it under `data/`.
2. Ask for a chart in the chat. Supported charts are line, bar/column,
   histogram, box plot, scatter, and numeric correlation heatmap.
3. Review the chart preview. Type `setuju` to save it to `outputs/charts`, or
   describe the requested revision.
4. Review the generated chart interpretation. Approve it or provide revised
   report text; approved reports are stored in `outputs/reports`.
5. Ask to `rangkum laporan` when ready. Approve or revise the synthesis to save
   it as `final_analyst.txt`.
6. Type `done` or `exit` to end the chat.
