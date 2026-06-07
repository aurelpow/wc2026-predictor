"""confederations.py — map national teams to their FIFA confederation and a
zone-strength score, so the model can value results by how tough the zone is.

Strength scores reflect that Europe (UEFA) and South America (CONMEBOL) are the
toughest zones; Oceania (OFC) the weakest. Tune `CONFED_STRENGTH` to taste.
Team names use the canonical spellings produced by data_loader.normalize_team
(i.e. the historical-results dataset's names).
"""
from __future__ import annotations

CONFED_STRENGTH = {
    "UEFA": 1.00,      # Europe
    "CONMEBOL": 1.00,  # South America
    "CAF": 0.72,       # Africa
    "AFC": 0.65,       # Asia
    "CONCACAF": 0.62,  # North/Central America & Caribbean
    "OFC": 0.38,       # Oceania
}
DEFAULT_STRENGTH = 0.62  # unknown / minor teams -> middle of the pack

_MEMBERS = {
    "CONMEBOL": [
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
        "Paraguay", "Peru", "Uruguay", "Venezuela",
    ],
    "UEFA": [
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
        "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
        "Czech Republic", "Denmark", "England", "Estonia", "Faroe Islands",
        "Finland", "France", "Georgia", "Germany", "Gibraltar", "Greece",
        "Hungary", "Iceland", "Republic of Ireland", "Ireland", "Israel", "Italy",
        "Kazakhstan", "Kosovo", "Latvia", "Liechtenstein", "Lithuania",
        "Luxembourg", "Malta", "Moldova", "Montenegro", "Netherlands",
        "North Macedonia", "Macedonia", "Northern Ireland", "Norway", "Poland",
        "Portugal", "Romania", "Russia", "San Marino", "Scotland", "Serbia",
        "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey",
        "Ukraine", "Wales",
    ],
    "CONCACAF": [
        "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados",
        "Belize", "Bermuda", "British Virgin Islands", "Canada", "Cayman Islands",
        "Costa Rica", "Cuba", "Curaçao", "Dominica", "Dominican Republic",
        "El Salvador", "Grenada", "Guatemala", "Guyana", "Haiti", "Honduras",
        "Jamaica", "Mexico", "Montserrat", "Nicaragua", "Panama", "Puerto Rico",
        "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines",
        "Sint Maarten", "Suriname", "Trinidad and Tobago",
        "Turks and Caicos Islands", "United States", "US Virgin Islands",
    ],
    "CAF": [
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
        "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros",
        "Congo", "DR Congo", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea",
        "Eswatini", "Swaziland", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
        "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya",
        "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco",
        "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda",
        "São Tomé and Príncipe", "Sao Tome and Principe", "Senegal", "Seychelles",
        "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan",
        "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe",
    ],
    "AFC": [
        "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan", "Brunei",
        "Cambodia", "China", "China PR", "Chinese Taipei", "Taiwan", "Guam",
        "Hong Kong", "India", "Indonesia", "Iran", "Iraq", "Japan", "Jordan",
        "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Macau", "Malaysia",
        "Maldives", "Mongolia", "Myanmar", "Nepal", "North Korea", "Oman",
        "Pakistan", "Palestine", "Philippines", "Qatar", "Saudi Arabia",
        "Singapore", "South Korea", "Sri Lanka", "Syria", "Tajikistan",
        "Thailand", "Timor-Leste", "Turkmenistan", "United Arab Emirates",
        "Uzbekistan", "Vietnam", "Yemen",
    ],
    "OFC": [
        "American Samoa", "Cook Islands", "Fiji", "New Caledonia", "New Zealand",
        "Papua New Guinea", "Samoa", "Solomon Islands", "Tahiti", "Tonga",
        "Vanuatu",
    ],
}

# team -> confederation code
TEAM_CONFED = {team: conf for conf, members in _MEMBERS.items() for team in members}


def confederation(team: str) -> str | None:
    return TEAM_CONFED.get(team)


def confed_strength(team: str) -> float:
    """Zone-strength score for a team (DEFAULT_STRENGTH if unknown)."""
    conf = TEAM_CONFED.get(team)
    return CONFED_STRENGTH.get(conf, DEFAULT_STRENGTH)
