# TindHire dataset builder

## How the dataset is built

To build the dataset, you need to have `resumes` folder in the `src` directory. This folder should contain two subfolders: `hired` and `rejected`. The hired folder should contain resumes that were hired and the rejected folder should contain resumes that were rejected.

1. Create a folder `src/resumes`
   ```
   resumes/
   ├─hired
   ├─rejected
   ```
   And put PDF files of valid or unvalid resumes accordingly
   This folder should NEVER be stored in git.

All PDF resumes in these folders will be converted into images and sent to the [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/) with [GPT-4 Turbo with Vision](https://learn.microsoft.com/en-us/azure/ai-services/openai/gpt-v-quickstart) API to extract the relevant data regarding some recrutment criterias. Azure service can't be called without authentication, to configure it, you have to create a `config.ini` file, in the `src/common` directory.

1. Create a `config.ini`
   ```
   [openai]
   key = your_api_key
   endpoint = your_azure_openai_endpoint
   deployment_name = your_deployment_name
   api_version = your_api_version
   ```
   Those information should NEVER be stored in git. 

The output format will be a JSON file for each resume that will be stored in `src/resumes/results`.


