import json

swagger = {
    "swagger": "2.0",
    "info": {
        "title": "ML Platform",
        "description": "API documentation for your application.",
        "version": "1.0"
    },
    "basePath": "/",
    "schemes": [
        "http"
    ],
    "paths": {
        "/upload": {
            "post": {
                "summary": "Upload a file and perform clustering analysis.",
                "parameters": [
                    {
                        "name": "file",
                        "in": "formData",
                        "type": "file",
                        "required": True,
                        "description": "The file to upload."
                    },
                    {
                        "name": "threshold",
                        "in": "formData",
                        "type": "number",
                        "required": True,
                        "description": "The threshold to identify useful columns for clustering."
                    },
                    {
                        "name": "algorithm",
                        "in": "formData",
                        "type": "string",
                        "required": True,
                        "description": "The clustering algorithm to use."
                    },
                    {
                        "name": "plot",
                        "in": "formData",
                        "type": "string",
                        "description": "Whether to generate a plot."
                    }
                ],
                "responses": {
                    "302": {
                        "description": "Redirects to the report page."
                    },
                    "404": {
                        "description": "Report not found."
                    }
                }
            }
        }
    }
}

with open('static/swagger.json', 'w') as outfile:
    json.dump(swagger, outfile)
