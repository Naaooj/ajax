from io import BytesIO
from openai import AzureOpenAI
from PIL import Image
import base64
import configparser
import fitz
import json
import os

def read_resume(pdf_path):
    # Open the pdf file
    pdf_document = fitz.open(pdf_path)
    dpi=96

    # List of images
    images = []

    # Iterate over all the pages in the PDF
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi) # Render the page to an image
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples) # Convert the image to a PIL-compatible format
        buffered = BytesIO()  # Create an in-memory bytes buffer
        image.save(buffered, format="PNG")  # Save the image to the buffer in PNG format
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')  # Encode the buffer content to base64
        images.append(img_str)  # Append the base64 string to the list

    return images

def main():
    # Convert the PDF to an image
    file_path = os.path.join(os.getcwd(), 'parser/resume.pdf')
    images = read_resume(file_path)

    # List to hold the user prompt parts
    user_prompt_parts = []

    for image in images:
        # Create the JSON object
        json_object = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image}"
            }
        }
    
        # Add the JSON object to the list
        user_prompt_parts.append(json_object)

    # Load configuration from config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Create an OpenAI client
    client = AzureOpenAI(
        api_key=config['openai']['key'],
        api_version=config['openai']['api_version'],
        base_url=f"{config['openai']['endpoint']}openai/deployments/{config['openai']['deployment_name']}",
    )

    #The type of data you might encounter in the resume includes but is not limited to: experience, previous experiences, education, skills, certifications, contact information, etc.

    user_prompt = """
    Extract the data in this resume, grouping data according to theme/sub groups, and then output into JSON.
    If there are blank data fields in the resume, please include them as "null" or "false" (if it's a boolean) values in the JSON object.
    Keep only 15 technologies, order them by years of experience descending
    Use the following structure:
    {
        "firstname": "",
        "lastname": "",
        "hasEuropeanNationality": true,
        "address": "",
        "email": "",
        "phoneNumber": "",
        "sex": "",
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

    system_prompt = """
    You are an AI assistant that extracts data from a resume and returns them as structured JSON objects. Do not return as a code block.
    """
    
    content = []
    content.append({
        "type": "text",
        "text": user_prompt
    })

    # Add the user prompt parts to the content list
    content.extend(user_prompt_parts)

    response = client.chat.completions.create(
        model=config['openai']['deployment_name'],
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=800,
        temperature=0.1
    )

    json_response = response.choices[0].message.content;
    print(json_response)

    # Save the response to a JSON file
    with open('output.json', 'w') as f:
        json.dump(json_response, f, indent=4)

if __name__ == '__main__':
    main()