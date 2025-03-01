"""
title: PlantUML diagram generator
author: crooy
description: This tool creates pretty diagram image from PlantUML code
author_url: https://github.com/crooy/openwebui-things
version: 0.1
required_open_webui_version: 0.5.0
requirements: plantuml
"""

import base64
from dataclasses import dataclass
from io import BytesIO
import textwrap
from PIL import Image
from fastapi import HTTPException, Request
import imghdr
import xml.etree.ElementTree as ET
from typing import Any, Awaitable, Callable, Optional, Tuple, Literal
from pydantic import BaseModel, Field
import requests


class Utils:
    @dataclass
    class EncodedImage:
        data: str
        size: Tuple[int, int]
        type: Literal["base64", "html", "component", "image"]

    # Scale down the image so that the shortest side is 768 pixels long
    @staticmethod
    def scale_down(width: int, height: int, shortest: int = 768, max_size: int = 2048):
        """
        Scales down the dimensions of an image while maintaining its aspect ratio.

        Parameters:
        width (int or str): The original width of the image.
        height (int or str): The original height of the image.

        Returns:
        tuple: The new width and height of the image, in the format (width, height). The dimensions are
            returned as int type.
        """
        # Calculate the ratio to scale the image while maintaining aspect ratio
        if width > height:
            ratio = max_size / float(width)
            max_height = int(height * ratio)
            max_width = max_size
        else:
            ratio = max_size / float(height)
            max_width = int(width * ratio)
            max_height = max_size

        # Scale down the image so that the shortest side is 768 pixels long
        if max_width < max_height:
            new_width = shortest
            new_height = int((new_width / max_width) * max_height)
        else:
            new_height = shortest
            new_width = int((new_height / max_height) * max_width)

        return new_width, new_height

    @staticmethod
    def encode_image(image):
        """base64 encode image and apply azure openai vision algo to calculate cost in tokens
        :param image: BytesIO buffer containing image bytes
        :return: tuple with base64 encoded image and cost in tokens"""
        # Open the image file
        image = Image.open(image)

        # Get the resolution of the image
        width, height = image.size

        # Scale down the image so that the shortest side is 768 pixels long
        new_width, new_height = Utils.scale_down(width, height)

        image = image.resize((new_width, new_height))

        # Convert the image back to bytes
        output = BytesIO()
        image.save(output, format="PNG")

        # Encode the image
        base64_string = base64.b64encode(output.getvalue()).decode("utf-8")

        return Utils.EncodedImage(base64_string, image.size, "base64")

    @staticmethod
    def encode_svg(svg):
        """base64 encode svg image and apply azure openai vision algo to calculate cost in tokens
        :param image: BytesIO buffer containing image bytes
        :return: tuple with base64 encoded image and cost in tokens"""
        # If the response is an SVG image, convert it to html
        root = ET.fromstring(svg.decode("utf-8"))
        width = int(round(float(root.get("width").replace("px", ""))))
        height = int(round(float(root.get("height").replace("px", ""))))

        # Scale down the image so that the shortest side is 768 pixels long
        new_width, new_height = Utils.scale_down(width, height)

        # Update the SVG XML with the new size
        root.set("width", str(new_width))
        root.set("height", str(new_height))
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        new_svg = ET.tostring(root, encoding="utf-8").decode("utf-8")

        # Encode the image
        base64_string = base64.b64encode(new_svg.encode("utf-8")).decode("utf-8")

        return Utils.EncodedImage(base64_string, (new_width, new_height), "svg")

    @staticmethod
    def get_image_html(image: EncodedImage):
        if image.type == "svg":
            image_type = "svg+xml"
        else:
            b = base64.b64decode(image.data)
            image_type = imghdr.what(None, h=b)
        html = r"""data:image/%s;base64,%s""" % (
            image_type,
            image.data,
        )
        return Utils.EncodedImage(html, image.size, "html")

    @staticmethod
    def get_image_js(image: EncodedImage):

        image = Utils.get_image_html(image)

        image_html = textwrap.dedent(
            r"""
        <div style="border:1px solid lightgrery;">
            <img src="%s" id="draggableImage"/>
        </div>
        """
            % (image.data,)
        )

        style = textwrap.dedent(
            r"""
        <style>
            #draggableImage {
                width: %s;
                touch-action: none;
                padding: 2px
            }
        </style>
        """
            % "100%"
        )

        script = textwrap.dedent(
            """
        <script src="https://cdn.jsdelivr.net/npm/interactjs@1.10.11/dist/interact.min.js"></script>
        <script>
            interact('#draggableImage')
                .draggable({
                    onmove: dragMoveListener,
                    onstart: function() { window.isDragging = true; },
                    onend: function() { window.isDragging = false; },
                    inertia: true,
                    autoScroll: true
                });

            var scale = 1;
            var x = 0;
            var y = 0;

            function dragMoveListener (event) {
                var target = event.target;
                x += event.dx;
                y += event.dy;

                target.style.webkitTransform =
                target.style.transform =
                    'translate(' + x + 'px, ' + y + 'px) scale(' + scale + ')';

                target.setAttribute('data-x', x);
                target.setAttribute('data-y', y);
            }

            document.getElementById('draggableImage').onwheel = function(e) {
                e.preventDefault();

                // Zoom in
                if (e.deltaY < 0) {
                    scale += 0.1;
                }
                // Zoom out
                else {
                    scale -= 0.1;
                }

                this.style.transform = 'translate(' + x + 'px, ' + y + 'px) scale(' + scale + ')';
            }
            document.getElementById('draggableImage').ondblclick = function(e) {
                // Create an HTML string with the image
                var html = '<html><body><img src="' + this.src + '"></body></html>';

                // Create a new Blob object using the HTML string
                var blob = new Blob([html], {type: 'text/html'});

                // Create a URL representing the Blob object
                var url = URL.createObjectURL(blob);

                // Open the URL in a new tab
                window.open(url, '_blank');
            }
        </script>
        """
        )
        html = image_html + style + script
        return Utils.EncodedImage(html, image.size, "html")

    @staticmethod
    def generate_plantuml_image(plantuml_server, data: str):
        # Send the PlantUML code to the server and get the resulting image
        try:
            response = requests.post(plantuml_server, data=data)
            # Check if the request was successful
            if response.status_code == 200:
                if plantuml_server.endswith("/svg"):
                    return Utils.get_image_html(Utils.encode_svg(response.content))
                    # return Utils.get_image_js(Utils.encode_svg(response.content))
                else:
                    return Utils.get_image_html(
                        Utils.encode_image(BytesIO(response.content))
                    )
                    # return Utils.get_image_js(Utils.encode_image(BytesIO(response.content)))
            else:
                error = f"PlantUML server error: {response.status_code}  \n{response.content.decode('utf-8')}"
                return error
        except Exception as e:
            error = f"PlantUML server error: {e}"
            return error


class Tools:
    class Valves(BaseModel):
        """Configuration valves for the PlantUML tool"""

        plantuml_server: str = Field(
            default="http://www.plantuml.com/plantuml/img/",
            description="PlantUML server URL for image generation",
        )

    def __init__(self) -> None:
        self.valves: Tools.Valves = self.Valves()

    def __getattr__(self, name: str) -> Any:
        """Handle dynamic attribute access"""
        return getattr(self.valves, name)

    async def generate_diagram(
        self,
        data: str,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Create a pretty diagram in PNG image format from PlantUML code

        :param data: block of valid PlantUML code
        :param __user__: user dictionary
        :param __event_emitter__: event emitter callback
        :return: Markdown image link
        """
        print("generating diagram using plantuml server:", self.valves.plantuml_server)
        print("data:", data)

        # Emit initial processing status
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Processing the PlantUML code...",
                        "done": False,
                    },
                }
            )

        if not data:
            return "Error: No PlantUML code provided"

        try:
            if not data.strip().startswith("@startuml"):
                data = "@startuml\n" + data
            if not data.strip().endswith("@enduml"):
                data = data + "\n@enduml"

            # if __event_emitter__:
            #     await __event_emitter__(
            #         {
            #             "type": "message",
            #             "data": {
            #                 "content": f"```plantuml\n{data}\n```\n"
            #             },
            #         }
            #     )
            # Use PlantUML library to get URL
            plantuml_image = Utils.generate_plantuml_image(
                self.valves.plantuml_server, data
            )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {
                            "content": f"![PlantUML Image]({plantuml_image.data})"
                        },
                    }
                )
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "The PlantUML diagram has been successfully created.",
                            "done": True,
                        },
                    }
                )
            return f"Your response must only contain the following code and only this code: ```plantuml\n{data}\n```\n"

        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error generating PlantUML URL: {str(e)}",
                            "done": True,
                        },
                    }
                )
            return f"There was an error in the PlantUML input code, please try again. Error: {str(e)}"
