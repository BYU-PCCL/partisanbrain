# enumerate all the categories in the NY Times dataset
# url: https://www.comparativeagendas.net/datasets_codebooks, under United States > New York Times Headlines
categories = {
    1: 'Macroeconomics',
    2: 'Civil Rights, Minority Issues, and Civil Liberties',
    3: 'Health',
    4: 'Agriculture',
    5: 'Labor',
    6: 'Education',
    7: 'Environment',
    8: 'Energy',
    9: 'Immigration',
    10: 'Transportation',
    12: 'Law, Crime, and Family Issues',
    13: 'Social Welfare',
    14: 'Community Development and Housing Issues',
    15: 'Banking, Finance, and Domestic Commerce',
    16: 'Defense',
    17: 'Space, Science, Technology and Communications',
    18: 'Foreign Trade',
    19: 'International Affairs and Foreign Aid',
    20: 'Government Operations',
    21: 'Public Lands and Water Management',
    24: 'State and Local Government Administration',
    26: 'Weather and Natural Disasters',
    27: 'Fires',
    28: 'Arts and Entertainment',
    29: 'Sports and Recreation',
    30: 'Death Notices',
    31: 'Churches and Religion',
    99: 'Other, Miscellaneous, and Human Interest',
}

category_descriptions = {
    'Macroeconomics': '''Examples: the administration's economic plans, economic conditions and issues, economic growth and 
outlook, state of the economy, long-term economic needs, recessions, general economic policy, 
promote economic recovery and full employment, demographic changes, population trends, recession 
effects on state and local economies, distribution of income, assuring an opportunity for employment 
to every American seeking work.''',
    'Civil Rights, Minority Issues, and Civil Liberties': '''Examples: Civil Rights Commission appropriations, civil rights violations, Civil Rights Act, Equal 
Rights amendments, equal employment opportunity laws, discrimination against women and 
minorities, appropriations for civil rights programs, civil rights enforcement, coverage of the civil 
rights act, employment discrimination involving several communities (age, gender, race, etc. in 
combination), taking private property, impact on private property rights, employment discrimination 
due to race, color, and religion.''',
    'Health': '''Examples: National Institute of Health (NIH) appropriations, Department of Health and Human 
Services (DHHS) appropriations, activities that provide little evidence of policy direction, 
commissions to study health issues, solvency of Medicare''',
    'Agriculture': '''Examples: DOA, USDA and FDA appropriations, general farm bills, farm legislation issues, economic 
conditions in agriculture, impact of budget reductions on agriculture, importance of agriculture to the 
U.S. economy, national farmland protection policies, agriculture and rural development appropriations, 
family farmers, state of American agriculture, farm program administration, long range agricultural 
policies, amend the Agriculture and Food Act, National Agricultural Bargaining Board.''',
    'Labor': '''Examples: Department of Labor budget requests and appropriations, assess change in labor markets to 
the year 2000, human resources development act, recent decline in the number of manufacturing jobs, 
national employment priorities, employment security administration financing, current labor market 
developments.''',
    'Education': '''Examples: Department of Education (DOEd) appropriations, state of education in the U.S., education 
programs development, education quality, national education methods, impact of education budget 
cuts, white house conference on education, National Institute of Education.''',
    'Environment': '''Examples: EPA, CEQ, ERDA budget requests and appropriations, implementation of the Clean Air 
Act, review of EPA regulations, Environmental Crimes Act, U.S. policies and international 
environmental issues, requirements for states to provide source pollution management programs, EPA 
pollution control programs, Comprehensive Environmental Response Act, environmental implications 
of the new energy act, environmental protection and energy conservation, adequacy of EPA budget and 
staff for implementing pollution control legislation.''',
    'Energy': '''Examples: Department of Energy (DOE) budget requests and appropriations, DOE and NRC budget 
requests and appropriations, national energy security policy, U.S. energy goals, U.S. energy supply and 
conservation, regulation of natural gas and electricity, impact of taxation on national energy policy, 
global energy needs, emergency plans for energy shortages, promotion of energy development 
projects, long-range energy needs of the U.S., energy capital requirements, establish the DOE, energy 
advisory committees.''',
    'Immigration': '''Examples: immigration of Cuban refugees to the U.S., refugee resettlement appropriations, HHS 
authority over immigration and public health, INS enforcement of immigration laws, legalization 
procedures for illegal immigrants, assessment of Haitian refugee detention by the U.S., immigration and 
education issues for aliens, adjusting visa allocations based on applicant job skills, DOL certification 
process for foreign engineers working in the U.S., denial of visas to political refugees, appropriations 
for the INS, citizenship issues, expedited citizenship for military service.''',
    'Transportation': '''Examples: Department of Transportation (DOT) and National Transportation Safety Board (NTSB) 
requests and appropriations, budget requests and appropriations for multiple agencies (NTSB, FAA, 
CAB), surface transportation programs, national transportation policy, rural transportation needs, 
adequacy of transportation systems, Interstate Commerce Commission policies and procedures, impact 
of budget cuts on DOT programs, highway and mass transit programs, transportation assistance 
programs, high-speed ground transportation systems.''',
    'Law, Crime, and Family Issues': '''Examples: emerging criminal justice issues, administration of criminal justice, revision of the criminal 
justice system, role of the U.S. commissioner in the criminal justice system.''',
    'Social Welfare': '''Examples: Health and Human Services (HHS) and Health, Education and Welfare (HEW) 
appropriations and budget requests, administration's welfare reform proposals, effectiveness of federal 
and state public welfare programs, social services proposals, public assistance programs, effects of 
economic and social deprivation on the psychology of underprivileged persons, social security and 
welfare benefits reforms.''',
    'Community Development and Housing Issues': '''Examples: Housing and Urban Development (HUD) budget requests and appropriations, housing and 
the housing market, HUD policy goals, building construction standards, future of the housing industry, 
national housing assistance legislation, administration and operation of national housing programs, 
housing safety standards.''',
    'Banking, Finance, and Domestic Commerce': '''Examples: Department of Commerce (DOC) and National Bureau of Standards (NBS) budget requests 
and appropriations, financial system structure and regulation, DOC reorganization plan, national 
materials policy, regulatory sunshine act, federal regulation of the economy, Interstate Commerce Act.''',
    'Defense': '''Examples: Department of Defense budget requests and appropriations (DOD), Department of the Air 
Force, Army, or Navy appropriations, armed services bills covering multiple subtopics, DOD 
operations and maintenance, defense production act, reorganization of the DOD, status of the national 
military establishment, establishment of the DOD, funding for defense activities of DOE, termination 
or designation of special defense areas.''',
    'Space, Science, Technology and Communications': '''Examples: Federal Communications Commission (FCC) and the Office of Science and Technology 
Policy budget requests and appropriations, science and engineering personnel requirements for the 
1990s, U.S. technology policy, FCC oversight review, reorganization of the FCC, national engineering 
and science policy, automation and technological change, FCC regulation of multiple subtopics (TV, 
telephone, cable, etc.).''',
    'Foreign Trade': '''Examples: Federal Trade Commission (FTC), U.S. International Trade Commission, International 
Trade Administration, or U.S. Custom Service budget requests and appropriations, world steel trade 
trends and structures, various tariff and trade bills, oversight hearings on U.S. foreign trade policy, 
U.S. trade relations with socialist economies, trade reform act, trade expansion act, tax and trade 
regulations, customs court issues, trading with enemy acts .''',
    'International Affairs and Foreign Aid': '''Examples: Department of State and U.S. Information Agency Budget Requests and Appropriations, 
U.S. foreign policy in view of recent world political developments, U.S. post cold war foreign policy, 
U.S. foreign policy and national defense issues, international tax treaties, international development 
and security, the U.S. ideological offensive--changing foreign opinion about the U.S., role of the 
diplomatic corps in foreign policy development and administration, foreign operations appropriations, 
information and educational exchange act, require Senate approval of treaty termination, establish the 
U.S. academy of peace, role of multinational corporations in U.S. foreign policy, Department of Peace, 
National Peace Agency.''',
    'Government Operations': '''Examples: budget requests for various agencies and independent commissions, budget requests for 
DOL, HHS, and DOE, appropriations for VA, HUD, and independent agencies, budget requests for 
DOC, DOS, and DOJ, appropriations for the GSA, budget requests for legislative branch programs, 
supplemental appropriation bills, appropriations for the Treasury, Postal Service, and general 
government appropriations.''',
    'Public Lands and Water Management': '''Examples: Budget Requests and Appropriations for the Department of Interior (DOI) and the Bureau 
of Land Management, proposed plan for the Department of Natural Resources, earth resources and 
drilling technology, resources planning, resource recovery act, activities and programs of the DOI, 
conveyance of certain real property of the U.S. government, conveyance of certain real property to 
states.''',
    'Arts and Entertainment': '''Examples: book, movie, music, art, and theater reviews, news about entertainment figures, hobbies 
(chess, bridge, fishing), cooking, restaurant reviews, interviews with chefs, travel stories, fashion 
stories, architecture, home improvement, gardening, charities, fund-raising events.''',
    'State and Local Government Administration': '''Examples: state and local candidates, campaigns, and elections, budget and tax issues, ethical issues 
about state and local officials, state and local buildings, museums, parks, landmarks, historical 
locations, state and local procurement and contracts, urban planning (zoning, land use, competition 
between cities to attract businesses, city boundaries), state and local services (water supply, street 
cleaning), constitutional issues (city charter revision), state and local statutes and ordinances, 
legislative action, speeches by the mayor or governor (inaugural, state of the city, state of the state 
addresses), partisan politics in the legislative arena, nominations to the state supreme court.''',
    'Weather and Natural Disasters': '''Examples: weather, natural disasters, natural events,
tsunamis, tornadoes, tornadoes.''',
    'Fires': '''Examples: fires, forest fires, major power outages, explosion of an overturned tanker truck, oil refinery.''',
    'Sports and Recreation': '''Examples: sports, sports events, sports stories, sports books, sports radio.''',
    'Death Notices': '''Examples: obituaries, death notices, national and state deaths.''',
    'Education': '''Examples: education, school, school education, education policy, school education, 
education and education policy, school education.''',
    'Churches and Religion': '''Examples: churches, religion, religious scandals, religious announcements.''',
    'Other, Miscellaneous, and Human Interest': '''For example, a story about a man stuck in his apartment’s elevator for three days. For example, a story about a man stuck in his apartment’s elevator for three days. For example, a story about how the Atlanta Zoo renting two panda bears from the Chinese 
government at a tab of $2 million a year (total, not per panda).''',
}