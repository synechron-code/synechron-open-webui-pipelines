"""
title: PlantUML diagram generator
author: david.sewell
description: Create diagram image from PlantUML code using PlantUML stdlib includes with syntax like !include <C4/C4_Container>
author_url: https://github.com/synechron-code/nexis-pipelines/plantuml-diagrams.py
version: 0.1
required_open_webui_version: 0.5.0
"""

import base64
from dataclasses import dataclass
from io import BytesIO
import logging
import json
import textwrap
from PIL import Image
from fastapi import HTTPException, Request
import imghdr
import xml.etree.ElementTree as ET
from typing import Any, Awaitable, Callable, Optional, Tuple, Literal
from pydantic import BaseModel, Field
import requests
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    def generate_plantuml_image(plantuml_server, data: str):
        # Send the PlantUML code to the server and get the resulting image
        try:
            response = requests.post(plantuml_server, data=data)
            # Check if the request was successful
            if response.status_code in [200, 400]:
                if plantuml_server.endswith("/svg"):
                    return Utils.get_image_html(Utils.encode_svg(response.content))
                    # return Utils.get_image_js(Utils.encode_svg(response.content))
                else:
                    return Utils.get_image_html(
                        Utils.encode_image(BytesIO(response.content))
                    )
                    # return Utils.get_image_js(Utils.encode_image(BytesIO(response.content)))
            else:
                errmsg = f"PlantUML server error: {response.status_code}  \n{response.content.decode('utf-8')}"
                error_trace = traceback.format_exc()
                logger.debug(f"{errmsg}\nTraceback: {error_trace}")
                return Utils.EncodedImage(errmsg, None, "error")
        except Exception as e:
            errmsg = f"PlantUML server error: {e}"
            error_trace = traceback.format_exc()
            logger.debug(f"{errmsg}\nTraceback: {error_trace}")
            return Utils.EncodedImage(errmsg, None, "error")

class Tools:
    class Valves(BaseModel):
        """Configuration valves for the PlantUML tool"""
        plantuml_server: str = Field(
            default="http://www.plantuml.com/plantuml/img/",
            description="PlantUML server URL for image generation",
        )
        enable_debug: bool = Field(
            default=False,
            description="Enable debug logging"
        )

    def __init__(self) -> None:
        self.valves: Tools.Valves = self.Valves()
        if self.valves.enable_debug:
            logger.setLevel("DEBUG")

    def __getattr__(self, name: str) -> Any:
        """Handle dynamic attribute access"""
        return getattr(self.valves, name)

    async def generate_diagram(
        self,
        data: str,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Generate a PlantUML diagram from PlantUML code.
        PlantUML code must comply with the PlantUML coding standards.
        PlantUML code must compile successfully on a PlantUML server.
        PlantUML code must only use includes from the following list of stdlib includes:

        list of stdlib includes:
        !include <C4/C4_Context>
        !include <C4/C4_Container>
        !include <C4/C4_Component>
        !include <C4/C4_Container>
        !include <C4/C4_Deployment>
        !include <C4/C4_Dynamic>
        !include <C4/C4_Sequence>

        :param data: block of valid PlantUML code that will compile successfully on a PlantUML server.
        :param __event_emitter__: event emitter callback
        :return: JSON object {"plantuml_code": code, "status": status}
        """
        self.emitter = __event_emitter__
        logger.info(f"generating diagram using plantuml server: {self.valves.plantuml_server}")
        logger.info(f"plantuml code:\n{data}")

        if not data:
            self._fail("No PlantUML code provided")

        # Emit initial processing status
        if self.emitter:
            await self.emitter(
                {
                    "type": "status",
                    "data": {
                        "description": "Processing the PlantUML code...",
                        "done": False,
                    },
                }
            )

        try:
            if not data.strip().startswith("@startuml"):
                data = "@startuml\n" + data
            if not data.strip().endswith("@enduml"):
                data = data + "\n@enduml"

            # Use PlantUML library to get URL
            plantuml_image = Utils.generate_plantuml_image(
                self.valves.plantuml_server, data
            )

            if plantuml_image.type == "error":
                errmsg = f"**ERROR:** {plantuml_image.data}"
                error_trace = traceback.format_exc()
                logger.exception(f"Traceback: {error_trace}")
                self._fail(errmsg)

            if self.emitter:
                await self.emitter(
                    {
                        "type": "message",
                        "data": {
                            "content": f"![PlantUML Image]({plantuml_image.data})"
                        },
                    }
                )
            if self.emitter:
                await self.emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": "The PlantUML diagram has been successfully created.",
                            "done": True,
                        },
                    }
                )
            return json.dumps(
                {
                    "plantuml_code": f"{data}",
                    "status": "done",
                },
                ensure_ascii=False,
            )

        except Exception as e:
            errmsg = f"Error generating PlantUML: {str(e)}"
            error_trace = traceback.format_exc()
            logger.exception(f"Traceback: {error_trace}")
            self._fail(errmsg)

    async def _fail(self, errmsg: str):
        if self.emitter:
            await self.emitter(
                {
                    "type": "status",
                    "data": {
                        "description": errmsg,
                        "done": True,
                    },
                }
            )
        logger.error(errmsg)
        return {"status": "error", "output": errmsg}


if __name__ == "__main__":
    tool = Tools()

# Examples for docstring if needed
"""
        Example 1:
        @startuml
        !include <C4/C4_Context>

        LAYOUT_WITH_LEGEND()

        title System Context diagram for Internet Banking System

        Person(customer, "Personal Banking Customer", "A customer of the bank, with personal bank accounts.")
        System(banking_system, "Internet Banking System", "Allows customers to view information about their bank accounts, and make payments.")

        System_Ext(mail_system, "E-mail system", "The internal Microsoft Exchange e-mail system.")
        System_Ext(mainframe, "Mainframe Banking System", "Stores all of the core banking information about customers, accounts, transactions, etc.")

        Rel(customer, banking_system, "Uses")
        Rel_Back(customer, mail_system, "Sends e-mails to")
        Rel_Neighbor(banking_system, mail_system, "Sends e-mails", "SMTP")
        Rel(banking_system, mainframe, "Uses")
        @enduml

        Example 2:
        @startuml
        !include <C4/C4_Container>

        ' LAYOUT_TOP_DOWN()
        ' LAYOUT_AS_SKETCH()
        LAYOUT_WITH_LEGEND()

        title Container diagram for Internet Banking System

        Person(customer, Customer, "A customer of the bank, with personal bank accounts")

        System_Boundary(c1, "Internet Banking") {
            Container(web_app, "Web Application", "Java, Spring MVC", "Delivers the static content and the Internet banking SPA")
            Container(spa, "Single-Page App", "JavaScript, Angular", "Provides all the Internet banking functionality to customers via their web browser")
            Container(mobile_app, "Mobile App", "C#, Xamarin", "Provides a limited subset of the Internet banking functionality to customers via their mobile device")
            ContainerDb(database, "Database", "SQL Database", "Stores user registration information, hashed auth credentials, access logs, etc.")
            Container(backend_api, "API Application", "Java, Docker Container", "Provides Internet banking functionality via API")
        }

        System_Ext(email_system, "E-Mail System", "The internal Microsoft Exchange system")
        System_Ext(banking_system, "Mainframe Banking System", "Stores all of the core banking information about customers, accounts, transactions, etc.")

        Rel(customer, web_app, "Uses", "HTTPS")
        Rel(customer, spa, "Uses", "HTTPS")
        Rel(customer, mobile_app, "Uses")

        Rel_Neighbor(web_app, spa, "Delivers")
        Rel(spa, backend_api, "Uses", "async, JSON/HTTPS")
        Rel(mobile_app, backend_api, "Uses", "async, JSON/HTTPS")
        Rel_Back_Neighbor(database, backend_api, "Reads from and writes to", "sync, JDBC")

        Rel_Back(customer, email_system, "Sends e-mails to")
        Rel_Back(email_system, backend_api, "Sends e-mails using", "sync, SMTP")
        Rel_Neighbor(backend_api, banking_system, "Uses", "sync/async, XML/HTTPS")
        @enduml

        Example 3:
        @startuml
        !include <C4/C4_Component>

        LAYOUT_WITH_LEGEND()

        title Component diagram for Internet Banking System - API Application

        Container(spa, "Single Page Application", "javascript and angular", "Provides all the internet banking functionality to customers via their web browser.")
        Container(ma, "Mobile App", "Xamarin", "Provides a limited subset ot the internet banking functionality to customers via their mobile mobile device.")
        ContainerDb(db, "Database", "Relational Database Schema", "Stores user registration information, hashed authentication credentials, access logs, etc.")
        System_Ext(mbs, "Mainframe Banking System", "Stores all of the core banking information about customers, accounts, transactions, etc.")

        Container_Boundary(api, "API Application") {
            Component(sign, "Sign In Controller", "MVC Rest Controller", "Allows users to sign in to the internet banking system")
            Component(accounts, "Accounts Summary Controller", "MVC Rest Controller", "Provides customers with a summary of their bank accounts")
            Component(security, "Security Component", "Spring Bean", "Provides functionality related to singing in, changing passwords, etc.")
            Component(mbsfacade, "Mainframe Banking System Facade", "Spring Bean", "A facade onto the mainframe banking system.")

            Rel(sign, security, "Uses")
            Rel(accounts, mbsfacade, "Uses")
            Rel(security, db, "Read & write to", "JDBC")
            Rel(mbsfacade, mbs, "Uses", "XML/HTTPS")
        }

        Rel(spa, sign, "Uses", "JSON/HTTPS")
        Rel(spa, accounts, "Uses", "JSON/HTTPS")

        Rel(ma, sign, "Uses", "JSON/HTTPS")
        Rel(ma, accounts, "Uses", "JSON/HTTPS")
        @enduml

        Example 4:
        @startuml
        !include <C4/C4_Context>

        'LAYOUT_TOP_DOWN()
        'LAYOUT_AS_SKETCH()
        LAYOUT_WITH_LEGEND()

        title System Landscape diagram for Big Bank plc

        Person(customer, "Personal Banking Customer", "A customer of the bank, with personal bank accounts.")

        Enterprise_Boundary(c0, "Big Bank plc") {
            System(banking_system, "Internet Banking System", "Allows customers to view information about their bank accounts, and make payments.")

            System_Ext(atm, "ATM", "Allows customers to withdraw cash.")
            System_Ext(mail_system, "E-mail system", "The internal Microsoft Exchange e-mail system.")

            System_Ext(mainframe, "Mainframe Banking System", "Stores all of the core banking information about customers, accounts, transactions, etc.")

            Person_Ext(customer_service, "Customer Service Staff", "Customer service staff within the bank.")
            Person_Ext(back_office, "Back Office Staff", "Administration and support staff within the bank.")
        }

        Rel_Neighbor(customer, banking_system, "Uses")
        Rel_R(customer, atm, "Withdraws cash using")
        Rel_Back(customer, mail_system, "Sends e-mails to")

        Rel_R(customer, customer_service, "Asks questions to", "Telephone")

        Rel_D(banking_system, mail_system, "Sends e-mail using")
        Rel_R(atm, mainframe, "Uses")
        Rel_R(banking_system, mainframe, "Uses")
        Rel_D(customer_service, mainframe, "Uses")
        Rel_U(back_office, mainframe, "Uses")

        Lay_D(atm, banking_system)

        Lay_D(atm, customer)
        Lay_U(mail_system, customer)
        @enduml

        Example 5:
        @startuml
        !include <C4/C4_Dynamic>

        LAYOUT_WITH_LEGEND()

        ContainerDb(c4, "Database", "Relational Database Schema", "Stores user registration information, hashed authentication credentials, access logs, etc.")
        Container(c1, "Single-Page Application", "JavaScript and Angular", "Provides all of the Internet banking functionality to customers via their web browser.")
        Container_Boundary(b, "API Application") {
        Component(c3, "Security Component", "Spring Bean", "Provides functionality Related to signing in, changing passwords, etc.")
        Component(c2, "Sign In Controller", "Spring MVC Rest Controller", "Allows users to sign in to the Internet Banking System.")
        }
        Rel_R(c1, c2, "Submits credentials to", "JSON/HTTPS")
        Rel(c2, c3, "Calls isAuthenticated() on")
        Rel_R(c3, c4, "select * from users where username = ?", "JDBC")
        @enduml

        Example 6:
        @startuml
        !include <C4/C4_Sequence>

        Container(c1, "Single-Page Application", "JavaScript and Angular", "Provides all of the Internet banking functionality to customers via their web browser.")

        Container_Boundary(b, "API Application")
        Component(c2, "Sign In Controller", "Spring MVC Rest Controller", "Allows users to sign in to the Internet Banking System.")
        Component(c3, "Security Component", "Spring Bean", "Provides functionality Related to signing in, changing passwords, etc.")
        Boundary_End()

        ContainerDb(c4, "Database", "Relational Database Schema", "Stores user registration information, hashed authentication credentials, access logs, etc.")

        Rel(c1, c2, "Submits credentials to", "JSON/HTTPS")
        Rel(c2, c3, "Calls isAuthenticated() on")
        Rel(c3, c4, "select * from users where username = ?", "JDBC")

        SHOW_LEGEND()
        @enduml

        Example 7:
        @startuml
        !include <C4/C4_Deployment>

        AddElementTag("fallback", $bgColor="#c0c0c0")
        AddRelTag("fallback", $textColor="#c0c0c0", $lineColor="#438DD5")

        ' calculated legend is used (activated in last line)
        ' LAYOUT_WITH_LEGEND()

        title Deployment Diagram for Internet Banking System - Live

        Deployment_Node(plc, "Big Bank plc", "Big Bank plc data center"){
            Deployment_Node(dn, "bigbank-api***\tx8", "Ubuntu 16.04 LTS"){
                Deployment_Node(apache, "Apache Tomcat", "Apache Tomcat 8.x"){
                    Container(api, "API Application", "Java and Spring MVC", "Provides Internet Banking functionality via a JSON/HTTPS API.")
                }
            }
            Deployment_Node(bigbankdb01, "bigbank-db01", "Ubuntu 16.04 LTS"){
                Deployment_Node(oracle, "Oracle - Primary", "Oracle 12c"){
                    ContainerDb(db, "Database", "Relational Database Schema", "Stores user registration information, hashed authentication credentials, access logs, etc.")
                }
            }
            Deployment_Node(bigbankdb02, "bigbank-db02", "Ubuntu 16.04 LTS", $tags="fallback") {
                Deployment_Node(oracle2, "Oracle - Secondary", "Oracle 12c", $tags="fallback") {
                    ContainerDb(db2, "Database", "Relational Database Schema", "Stores user registration information, hashed authentication credentials, access logs, etc.", $tags="fallback")
                }
            }
            Deployment_Node(bb2, "bigbank-web***\tx4", "Ubuntu 16.04 LTS"){
                Deployment_Node(apache2, "Apache Tomcat", "Apache Tomcat 8.x"){
                    Container(web, "Web Application", "Java and Spring MVC", "Delivers the static content and the Internet Banking single page application.")
                }
            }
        }

        Deployment_Node(mob, "Customer's mobile device", "Apple IOS or Android"){
            Container(mobile, "Mobile App", "Xamarin", "Provides a limited subset of the Internet Banking functionality to customers via their mobile device.")
        }

        Deployment_Node(comp, "Customer's computer", "Microsoft Windows or Apple macOS"){
            Deployment_Node(browser, "Web Browser", "Google Chrome, Mozilla Firefox, Apple Safari or Microsoft Edge"){
                Container(spa, "Single Page Application", "JavaScript and Angular", "Provides all of the Internet Banking functionality to customers via their web browser.")
            }
        }

        Rel(mobile, api, "Makes API calls to", "json/HTTPS")
        Rel(spa, api, "Makes API calls to", "json/HTTPS")
        Rel_U(web, spa, "Delivers to the customer's web browser")
        Rel(api, db, "Reads from and writes to", "JDBC")
        Rel(api, db2, "Reads from and writes to", "JDBC", $tags="fallback")
        Rel_R(db, db2, "Replicates data to")

        SHOW_LEGEND()
        @enduml

        Example 8:
        @startuml
        !include <C4/C4_Deployment>

        AddElementTag("fallback", $bgColor="#c0c0c0")
        AddRelTag("fallback", $textColor="#c0c0c0", $lineColor="#438DD5")

        WithoutPropertyHeader()

        ' calculated legend is used (activated in last line)
        ' LAYOUT_WITH_LEGEND()

        title Deployment Diagram for Internet Banking System - Live

        Deployment_Node(plc, "Live", "Big Bank plc", "Big Bank plc data center"){
            AddProperty("Location", "London and Reading")
            Deployment_Node_L(dn, "bigbank-api***\tx8", "Ubuntu 16.04 LTS", "A web server residing in the web server farm, accessed via F5 BIG-IP LTMs."){
                AddProperty("Java Version", "8")
                AddProperty("Xmx", "512M")
                AddProperty("Xms", "1024M")
                Deployment_Node_L(apache, "Apache Tomcat", "Apache Tomcat 8.x", "An open source Java EE web server."){
                    Container(api, "API Application", "Java and Spring MVC", "Provides Internet Banking functionality via a JSON/HTTPS API.")
                }
            }
            AddProperty("Location", "London")
            Deployment_Node_L(bigbankdb01, "bigbank-db01", "Ubuntu 16.04 LTS", "The primary database server."){
                Deployment_Node_L(oracle, "Oracle - Primary", "Oracle 12c", "The primary, live database server."){
                    ContainerDb(db, "Database", "Relational Database Schema", "Stores user registration information, hashed authentication credentials, access logs, etc.")
                }
            }
            AddProperty("Location", "Reading")
            Deployment_Node_R(bigbankdb02, "bigbank-db02", "Ubuntu 16.04 LTS", "The secondary database server.", $tags="fallback") {
                Deployment_Node_R(oracle2, "Oracle - Secondary", "Oracle 12c", "A secondary, standby database server, used for failover purposes only.", $tags="fallback") {
                    ContainerDb(db2, "Database", "Relational Database Schema", "Stores user registration information, hashed authentication credentials, access logs, etc.", $tags="fallback")
                }
            }
            AddProperty("Location", "London and Reading")
            Deployment_Node_R(bb2, "bigbank-web***\tx4", "Ubuntu 16.04 LTS", "A web server residing in the web server farm, accessed via F5 BIG-IP LTMs."){
                AddProperty("Java Version", "8")
                AddProperty("Xmx", "512M")
                AddProperty("Xms", "1024M")
                Deployment_Node_R(apache2, "Apache Tomcat", "Apache Tomcat 8.x", "An open source Java EE web server."){
                    Container(web, "Web Application", "Java and Spring MVC", "Delivers the static content and the Internet Banking single page application.")
                }
            }
        }

        Deployment_Node(mob, "Customer's mobile device", "Apple IOS or Android"){
            Container(mobile, "Mobile App", "Xamarin", "Provides a limited subset of the Internet Banking functionality to customers via their mobile device.")
        }

        Deployment_Node(comp, "Customer's computer", "Microsoft Windows of Apple macOS"){
            Deployment_Node(browser, "Web Browser", "Google Chrome, Mozilla Firefox, Apple Safari or Microsoft Edge"){
                Container(spa, "Single Page Application", "JavaScript and Angular", "Provides all of the Internet Banking functionality to customers via their web browser.")
            }
        }

        Rel(mobile, api, "Makes API calls to", "json/HTTPS")
        Rel(spa, api, "Makes API calls to", "json/HTTPS")
        Rel_U(web, spa, "Delivers to the customer's web browser")
        Rel(api, db, "Reads from and writes to", "JDBC")
        Rel(api, db2, "Reads from and writes to", "JDBC", $tags="fallback")
        Rel_R(db, db2, "Replicates data to")

        SHOW_LEGEND()
        @enduml

        Example 9:
        @startuml
        !include <C4/C4_Container>

        SHOW_PERSON_OUTLINE()
        AddElementTag("backendContainer", $fontColor=$ELEMENT_FONT_COLOR, $bgColor="#335DA5", $shape=EightSidedShape(), $legendText="backend container\neight sided")
        AddRelTag("async", $textColor=$ARROW_FONT_COLOR, $lineColor=$ARROW_COLOR, $lineStyle=DashedLine())
        AddRelTag("sync/async", $textColor=$ARROW_FONT_COLOR, $lineColor=$ARROW_COLOR, $lineStyle=DottedLine())

        title Container diagram for Internet Banking System

        Person(customer, Customer, "A customer of the bank, with personal bank accounts")

        System_Boundary(c1, "Internet Banking") {
            Container(web_app, "Web Application", "Java, Spring MVC", "Delivers the static content and the Internet banking SPA")
            Container(spa, "Single-Page App", "JavaScript, Angular", "Provides all the Internet banking functionality to customers via their web browser")
            Container(mobile_app, "Mobile App", "C#, Xamarin", "Provides a limited subset of the Internet banking functionality to customers via their mobile device")
            ContainerDb(database, "Database", "SQL Database", "Stores user registration information, hashed auth credentials, access logs, etc.")
            Container(backend_api, "API Application", "Java, Docker Container", "Provides Internet banking functionality via API", $tags="backendContainer")
        }

        System_Ext(email_system, "E-Mail System", "The internal Microsoft Exchange system")
        System_Ext(banking_system, "Mainframe Banking System", "Stores all of the core banking information about customers, accounts, transactions, etc.")

        Rel(customer, web_app, "Uses", "HTTPS")
        Rel(customer, spa, "Uses", "HTTPS")
        Rel(customer, mobile_app, "Uses")

        Rel_Neighbor(web_app, spa, "Delivers")
        Rel(spa, backend_api, "Uses", "async, JSON/HTTPS", $tags="async")
        Rel(mobile_app, backend_api, "Uses", "async, JSON/HTTPS", $tags="async")
        Rel_Back_Neighbor(database, backend_api, "Reads from and writes to", "sync, JDBC")

        Rel_Back(customer, email_system, "Sends e-mails to")
        Rel_Back(email_system, backend_api, "Sends e-mails using", "sync, SMTP")
        Rel_Neighbor(backend_api, banking_system, "Uses", "sync/async, XML/HTTPS", $tags="sync/async")

        SHOW_LEGEND()
        @enduml

        Example 10:
        @startuml
        !include <C4/C4_Container>
        !define DEVICONS https://raw.githubusercontent.com/tupadr3/plantuml-icon-font-sprites/master/devicons
        !define FONTAWESOME https://raw.githubusercontent.com/tupadr3/plantuml-icon-font-sprites/master/font-awesome-5
        !include DEVICONS/angular.puml
        !include DEVICONS/dotnet.puml
        !include DEVICONS/java.puml
        !include DEVICONS/msql_server.puml
        !include FONTAWESOME/server.puml
        !include FONTAWESOME/envelope.puml

        ' LAYOUT_TOP_DOWN()
        ' LAYOUT_AS_SKETCH()
        LAYOUT_WITH_LEGEND()

        title Container diagram for Internet Banking System

        Person(customer, Customer, "A customer of the bank, with personal bank accounts")

        System_Boundary(c1, "Internet Banking") {
            Container(web_app, "Web Application", "Java, Spring MVC", "Delivers the static content and the Internet banking SPA", "java")
            Container(spa, "Single-Page App", "JavaScript, Angular", "Provides all the Internet banking functionality to customers via their web browser", "angular")
            Container(mobile_app, "Mobile App", "C#, Xamarin", "Provides a limited subset of the Internet banking functionality to customers via their mobile device", "dotnet")
            ContainerDb(database, "Database", "SQL Database", "Stores user registration information, hashed auth credentials, access logs, etc.", "mysql_server")
            Container(backend_api, "API Application", "Java, Docker Container", "Provides Internet banking functionality via API", "server")
        }

        System_Ext(email_system, "E-Mail System", "The internal Microsoft Exchange system", "envelope")
        System_Ext(banking_system, "Mainframe Banking System", "Stores all of the core banking information about customers, accounts, transactions, etc.")

        Rel(customer, web_app, "Uses", "HTTPS")
        Rel(customer, spa, "Uses", "HTTPS")
        Rel(customer, mobile_app, "Uses")

        Rel_Neighbor(web_app, spa, "Delivers")
        Rel(spa, backend_api, "Uses", "async, JSON/HTTPS")
        Rel(mobile_app, backend_api, "Uses", "async, JSON/HTTPS")
        Rel_Back_Neighbor(database, backend_api, "Reads from and writes to", "sync, JDBC")

        Rel_Back(customer, email_system, "Sends e-mails to")
        Rel_Back(email_system, backend_api, "Sends e-mails using", "sync, SMTP")
        Rel_Neighbor(backend_api, banking_system, "Uses", "sync/async, XML/HTTPS")
        @enduml
"""