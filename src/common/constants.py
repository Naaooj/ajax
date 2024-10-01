SYSTEM_PROMPT = """
    You are an AI assistant that extracts data from a resume and returns them as structured JSON objects. Do not return as a code block.
    """

USER_PROMPT = """
    Extract the data in this resume, grouping data according to theme/sub groups, and then output into JSON.
    If there are blank data fields in the resume, please include them as "null" or "false" (if it's a boolean) values in the JSON object.
    Keep only 20 technologies, order them by years of experience descending
    Use the following structure:
    {
        "hasEuropeanNationality": true,
        "totalYearsOfExperience": 0.0,
        "totalYearsOfStudies": 0.0,
        "diplomas": [{
            "level": "",
            "year": 0
        }],
        "motherTongue": "",
        "otherLanguages": [{
            "language": "",
            "proficiency": ""
        }],
        "technologies": [{
            "name": "",
            "yearsOfExperience": 0.0
        }]
    }
    """