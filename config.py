
class Config:
    """Flask configuration object."""
    
    # Flask
    FLASK_ENV = 'production'
 
    # Swagger
    SWAGGER = {
        'version': 'v1',
        'title': 'Intent detection API',
        'description': 'API offering access to an intent classification model',
        'termsOfService': 'Free to use',
        'basePath': '/',
        'specs': [
            {
                "endpoint": 'docs',
                "route": '/docs.json',
            }
        ]
    }
    