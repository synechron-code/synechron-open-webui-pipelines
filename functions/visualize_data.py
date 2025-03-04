"""
title: Make charts out of your data v2
author: David Sewell
author_url: https://github.com/syencrhon-code
funding_url: https://github.com/open-webui
version: 2.0.1
"""

import base64
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Awaitable, Callable
import os
from open_webui.models.files import Files, FileForm, FileMeta
from open_webui.config import (
    UPLOAD_DIR
)
import uuid
import logging
from openai import OpenAI
import time
import traceback

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT_BUILD_CHARTS = """

Objective:
Your goal is to read the query, extract the data, choose the appropriate chart to present the data, and produce the HTML to display it.

Steps:

	1.	Read and Examine the Query:
	•	Understand the user’s question and identify the data provided.
	2.	Analyze the Data:
	•	Examine the data in the query to determine the appropriate chart type (e.g., bar chart, pie chart, line chart) for effective visualization.
    •	determine the axis limits, axis steps, chart style, labels
    •	determine whether to combine items on a single chart or use separate charts
    •	consider if all data points will fit onto the chart if combining charts
	3.	Generate HTML:
	•	Create the HTML code to present the data using the selected chart format.
	4.	Handle No Data Situations:
	•	If there is no data in the query or the data cannot be presented as a chart, generate a humorous or funny HTML response indicating that the data cannot be presented.
    5.	Calibrate the chart scale based on the data:
	•	based on the data try to make the scale of the chart as readable as possible.

Key Considerations:

	-	Your output should only include HTML code, without any additional text.
    -   Generate only HTML. Do not include any additional words or explanations.
    -   Normalize the data using unit conversion so values are comparable.
    -   Sanitize the data by removing non alpha numeric from the data.
    -   Calibrate the chart scale based on the data for everything to be readable.
    -   Generate only html code , nothing else , only html.


Example1 :
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Chart</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="chart" style="width: 100%; height: 500px;"></div>
    <button id="save-button">Save Screenshot</button>
    <script>
        // Data for the chart
        var data = [{
            x: [''Category 1'', ''Category 2'', ''Category 3''],
            y: [20, 14, 23],
            type: ''bar''
        }];

        // Layout for the chart
        var layout = {
            title: ''Interactive Bar Chart'',
            xaxis: {
                title: ''Categories''
            },
            yaxis: {
                title: ''Values''
            }
        };

        // Render the chart
        Plotly.newPlot(''chart'', data, layout);

        // Function to save screenshot
        document.getElementById(''save-button'').onclick = function() {
            Plotly.downloadImage(''chart'', {format: ''png'', width: 800, height: 600, filename: ''chart_screenshot''});
        };

        // Function to update chart attributes
        function updateChartAttributes(newData, newLayout) {
            Plotly.react(''chart'', newData, newLayout);
        }

        // Example of updating chart attributes
        var newData = [{
            x: [''New Category 1'', ''New Category 2'', ''New Category 3''],
            y: [10, 22, 30],
            type: ''bar''
        }];

        var newLayout = {
            title: ''Updated Bar Chart'',
            xaxis: {
                title: ''New Categories''
            },
            yaxis: {
                title: ''New Values''
            }
        };

        // Call updateChartAttributes with new data and layout
        // updateChartAttributes(newData, newLayout);
    </script>
</body>
</html>


Example2:
<!DOCTYPE html>
<html>
<head>
    <title>Collaborateurs par Métier/Fonction</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="myChart" style="width: 100%; max-width: 700px; height: 500px; margin: 0 auto;"></div>
    <script>
        var data = [{
            x: ["Ingénieur Système", "Solution Analyst", "Ingénieur d''études et Développement", "Squad Leader", "Architecte d''Entreprise", "Tech Lead", "Architecte Technique", "Référent Méthodes / Outils"],
            y: [5, 3, 2, 1, 1, 1, 1, 1],
            type: "bar",
            marker: {
                color: "rgb(49,130,189)"
            }
        }];
        var layout = {
            title: "Collaborateurs de STT par Métier/Fonction",
            xaxis: {
                title: "Métier/Fonction"
            },
            yaxis: {
                title: "Nombre de Collaborateurs"
            }
        };
        Plotly.newPlot("myChart", data, layout);
    </script>
</body>
</html>

2.	No Data or Unchartable Data:
<html>
<body>
    <h1>We''re sorry, but your data can''t be charted.</h1>
    <p>Maybe try feeding it some coffee first?</p>
    <img src="https://media.giphy.com/media/l4EoTHjkw0XiYtNRG/giphy.gif" alt="Funny Coffee GIF">
</body>
</html>
"""

USER_PROMPT_GENERATE_HTML = """
Giving this query  {Query} generate the necessary html qurty.
"""

# Initialize OpenAI client


class Action:
    class Valves(BaseModel):
        show_status: bool = Field(default=True, description="Show status of the action.")
        html_filename: str = Field(
            default="json_visualizer.html",
            description="Name of the HTML file to be created or retrieved.",
        )
        OPENIA_KEY: str = Field(
            default="",
            description="key to consume OpenIA interface like LLM for example a litellm key.",
        )
        OPENIA_URL: str = Field(
            default="",
            description="Host where to consume the OpenIA interface like llm",
        )
        MODEL_NAME: str = Field(
            default="llama3.1:latest",
            description="model name",
        )
        DEBUG: bool = Field(
            default=True,
            description="Enable debug logging"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.openai = None
        self.html_content = """
        """
        if self.valves.DEBUG:
            logger.setLevel(logging.DEBUG)


    def create_or_get_file(self, user_id: str, html_data: str) -> str:

        filename = str(int(time.time() * 1000)) + self.valves.html_filename
        directory = "action_embed"

        logger.debug(f"Attempting to create or get file: {filename}")

        # Check if the file already exists
        existing_files = Files.get_files()
        for file in existing_files:
            if file.filename == f"{directory}/{user_id}/{filename}" and file.user_id == user_id:
                logger.debug(f"Existing file found. Updating content.")
                # Update the existing file with new JSON data
                self.update_html_content(file.meta["path"], html_data)
                return file.id

        # If the file doesn't exist, create it
        base_path = os.path.join(UPLOAD_DIR, directory)
        os.makedirs(base_path, exist_ok=True)
        file_path = os.path.join(base_path, filename)

        logger.debug(f"Creating new file at: {file_path}")
        self.update_html_content(file_path, html_data)

        file_id = str(uuid.uuid4())
        meta = FileMeta(
            name="Modern Data Visualizer",
            content_type="text/html",
            size=os.path.getsize(file_path)
        )

        # Create a new file entry
        file_data = FileForm(
            id=file_id,
            filename=f"{directory}/{user_id}/{filename}",
            path=file_path,
            meta=meta.model_dump(),
        )
        new_file = Files.insert_new_file(user_id, file_data)
        logger.debug(f"New file created with ID: {new_file.id}")
        return new_file.id

    def update_html_content(self, file_path: str, html_content: str):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.debug(f"HTML content updated at: {file_path}")

    def escape_html_content(self, html_content: str) -> str:
        """
        Escapes double quotes in the HTML content to make it safe for embedding in an iframe as data:text/html.
        """
        return html_content.replace('"', '&quot;')

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Optional[dict]:
        if self.valves.DEBUG:
            logger.setLevel(logging.DEBUG)

        logger.debug(f"action:{__name__} started")

        if self.valves.show_status and __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Analysing Data",
                        "done": False,
                    },
                }
            )

        try:
            original_content = body["messages"][-1]["content"]
            self.openai = OpenAI(api_key=self.valves.OPENIA_KEY, base_url=self.valves.OPENIA_URL)

            response = self.openai.chat.completions.create(
                model=self.valves.MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_BUILD_CHARTS},
                    {
                        "role": "user",
                        "content": USER_PROMPT_GENERATE_HTML.format(Query=body["messages"][-1]["content"]),
                    },
                ],
                max_tokens=1000,
                n=1,
                stop=None,
                temperature=0.7,
            )

            html_content = response.choices[0].message.content


            logger.debug("-----------------------------")
            # print html content in pretty and readable format
            # this is to help debug
            logger.debug(html_content)
            logger.debug("-----------------------------")

            user_id = __user__["id"]
            file_id = self.create_or_get_file(user_id, html_content)

            # Create the HTML embed tag
            html_embed_tag = f"{{{{HTML_FILE_ID_{file_id}}}}}"

            # If MarkdownTokens.svelt recognized `<iframe`` not just `<iframe src="${WEBUI_BASE_URL}/api/v1/files/` then this would work
            # html_content_base64 = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
            # html_embed_tag = f'<iframe src="{data:text/html;base64,{html_content_base64}}" width="100%" frameborder="0" onload="this.style.height=(this.contentWindow.document.body.scrollHeight+20)+px;"></iframe>'

            # Append the HTML embed tag to the original content on a new line
            body["messages"][-1]["content"] = f"{original_content}\n\n{html_embed_tag}"

            if self.valves.show_status and __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Visualise the chart",
                            "done": True,
                        },
                    }
                )
                logger.debug(f" objects visualized")

        except Exception as e:
            error_message = f"Error visualizing JSON: {str(e)}"
            logger.error(f"Error: {error_message}")
            error_trace = traceback.format_exc()
            logger.exception(f"Traceback: {error_trace}")
            body["messages"][-1]["content"] += f"\n\nError: {error_message}\n\nTraceback: {error_trace}"

            await __event_emitter__({"type": "message", "data": {"content": f"\n\nError: {error_message}"}})

            if self.valves.show_status and __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Error Visualizing JSON",
                            "done": True,
                        },
                    }
                )

        logger.debug(f"action:{__name__} completed")
        return body
