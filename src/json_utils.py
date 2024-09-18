import json

class JsonUtils:

    @staticmethod
    def flatten(json_file):
        with open(json_file, 'r') as file:
            content = json.load(file)

        # Extract basic information
        firstname = content.get('firstname', '')
        lastname = content.get('lastname', '')
        nationality = 'European' if content.get('hasEuropeanNationality', False) else 'Non-European'
        address = content.get('address', '')
        email = content.get('email', '')
        phone = content.get('phoneNumber', '')
        sex = content.get('sex', '')
        experience = content.get('totalYearsOfExperience', 0.0)
        studies = content.get('totalYearsOfStudies', 0.0)
        mother_tongue = content.get('motherTongue', '')

        # Extract diplomas
        diplomas_list = content.get('diplomas', [])
        if diplomas_list is None:
            diplomas_list = []
        diplomas = ', '.join([f"{d['level']} ({d['year']})" for d in diplomas_list])

        # Extract languages
        languages_list = content.get('otherLanguages', [])
        if languages_list is None:
            languages_list = []
        languages = ', '.join([f"{l['language']} ({l.get('proficiency')})" for l in languages_list if l and l.get('proficiency')])

        # Extract technologies
        technologies_list = content.get('technologies', [])
        if technologies_list is None:
            technologies_list = []
        technologies = ', '.join([f"{t['name']} ({t['yearsOfExperience']} years)" for t in technologies_list if t])

        return f"{firstname} {lastname} {nationality} {address} {email} {phone} {sex} {experience} years of experience {studies} years of studies {mother_tongue} {diplomas} {languages} {technologies}"
        