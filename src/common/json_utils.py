import json

class JsonUtils:

    @staticmethod
    def flatten(json_file):
        with open(json_file, 'r') as file:
            content = json.load(file)
        return JsonUtils.flatten_content(content)

    @staticmethod
    def flatten_content(json_content):
        # Extract basic information
        nationality = 'European' if json_content.get('hasEuropeanNationality', False) else 'Non-European'
        experience = json_content.get('totalYearsOfExperience', 0.0)
        studies = json_content.get('totalYearsOfStudies', 0.0)
        mother_tongue = json_content.get('motherTongue', '')

        # Extract diplomas
        diplomas_list = json_content.get('diplomas', [])
        if diplomas_list is None:
            diplomas_list = []
        diplomas = ', '.join([f"{d['level']} ({d['year']})" for d in diplomas_list])

        # Extract languages
        languages_list = json_content.get('otherLanguages', [])
        if languages_list is None:
            languages_list = []
        languages = ', '.join([f"{l['language']} ({l.get('proficiency')})" for l in languages_list if l and l.get('proficiency')])

        # Extract technologies
        technologies_list = json_content.get('technologies', [])
        if technologies_list is None:
            technologies_list = []
        technologies = ', '.join([f"{t['name']} ({t['yearsOfExperience']} years)" for t in technologies_list if t])

        return f"{nationality} {experience} years of experience {studies} years of studies {mother_tongue} {diplomas} {languages} {technologies}"
