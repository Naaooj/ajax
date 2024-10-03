# AJAX dataset builder

## How the dataset is built

To build the dataset, you need to have `resumes` folder in the `src` directory (as described in the main [README.md](../../README.md)). This folder should contain two subfolders: `hired` and `rejected`. The hired folder should contain resumes that were hired and the rejected folder should contain resumes that were rejected.

All PDF resumes in these folders will be converted into images and sent to the [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/) with [GPT-4 Turbo with Vision](https://learn.microsoft.com/en-us/azure/ai-services/openai/gpt-v-quickstart) API to extract the relevant data regarding Pictet Technologies recrutment criterias.

The output format will be a JSON file for each resume that will be stored in `src/resumes/results`.


