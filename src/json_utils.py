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
        diplomas = ', '.join([f"{d['level']} ({d['year']})" for d in content.get('diplomas', [])])

        # Extract languages
        languages = ', '.join([f"{l['language']} ({l['proficiency']})" for l in content.get('otherLanguages', []) if l['proficiency']])

        # Extract technologies
        technologies = ', '.join([f"{t['name']} ({t['yearsOfExperience']} years)" for t in content.get('technologies', [])])

        return f"{firstname} {lastname} {nationality} {address} {email} {phone} {sex} {experience} years of experience {studies} years of studies {mother_tongue} {diplomas} {languages} {technologies}"
        