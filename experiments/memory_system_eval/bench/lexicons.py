"""Lexicons for MemBench-H (on-device assistant memory). All text is generated locally from these lists.

Pools of names and lexical values are split disjointly across train / val / test_id / test_ood (see gen_membench.split_pool).
Attributes carry: canonical name, training aliases, held-out (OOD) aliases whose novel words never occur in train/val text,
attribute-specific statement and question templates per split family, update templates, and a confuser attribute.
Placeholders: {N} entity name (or 'my' forms for the self entity: {M} sentence-initial 'My', {m} 'my'), {v} value, {old} old value,
{pos}/{sub}/{obj}/{Sub} pronouns of {N}, {alias}, {art} 'a'/'an' for the value."""

FEMALE = """Sara Emma Olivia Mia Sophia Isabella Charlotte Amelia Evelyn Abigail Emily Ella Elizabeth Camila Sofia Scarlett Madison
Aria Chloe Penelope Layla Lily Nora Zoey Hannah Audrey Bella Claire Lucy Anna Caroline Allison Maya Sarah Madelyn Adeline Alexa
Ariana Elena Gabriella Naomi Alice Hailey Eva Emilia Piper Cora Kaylee Lydia Aubrey Arianna Eliana Melanie Gianna Isabelle Julia
Valentina Clara Vivian Mackenzie Madeline Delilah Isla Katherine Sophie Josephine Liliana Maria Kylie Natalie Annabelle Alexandra
Ximena Ashley Brianna Mary Andrea Leilani Jasmine Lyla Margaret Alyssa Norah Khloe Kayla Eliza Rachel Esther Diana Fiona Helen
Irene Laura Monica Nina Paula Rita Tina Wendy Yvonne Zara Beatrice Carmen Daphne Edith Greta Hilda Ingrid Joan Karen Linda Martha
Nancy Olga Pamela Rebecca Susan Teresa Ursula Vera Agnes Bianca Celeste Doris Elsa Frida Gloria Heidi Judith Kathy Leah Miriam
Nadia Priya Rosa Selma Tara Uma Wanda Yasmin Lorena Marisol Fatima Aisha Keiko Yuki Mei Ling Anika Svetlana Natasha Ines Lucia
Renata Paola Simone Colette Brigitte Astrid Sigrid Freya Maren Elif Leyla Noor Zainab Amara Imani Nia Ayesha Sana Hina Mariko""".split()

MALE = """Liam Noah Elijah James William Benjamin Lucas Henry Theodore Jack Levi Alexander Jackson Mateo Daniel Michael Mason
Sebastian Ethan Logan Owen Samuel Jacob Asher Aiden John Joseph Wyatt David Leo Luke Julian Gabriel Grayson Isaac Lincoln Anthony
Hudson Dylan Ezra Thomas Charles Christopher Josiah Isaiah Andrew Elias Joshua Nathan Caleb Ryan Adrian Miles Eli Nolan Aaron
Cameron Ezekiel Colton Luca Landon Jonathan Santiago Axel Easton Jeremiah Roman Connor Jameson Robert Greyson Ian Carson
Leonardo Nicholas Dominic Everett Xavier Jose Adam Jace Wesley Kayden Silas Bennett Declan Weston Evan Emmett Micah Damian
Brayden Harrison Bryson Sawyer Amir Jason Giovanni Vincent Ayden Myles Diego Nathaniel Jonah Tyler Cole Braxton George Milo
Zachary Ashton Luis Jasper Kaiden Adriel Gavin Calvin Juan Maxwell Carlos Emmanuel Jayce Lorenzo Ivan Kevin Malachi Elliott Rhett
Arthur Victor Bruce Edgar Felix Gordon Harold Igor Kurt Lionel Marcus Neil Oscar Patrick Quentin Ralph Simon Trevor Walter Xander
Yusuf Zane Omar Hamza Tariq Kenji Hiroshi Takeshi Wei Jun Arjun Rohan Vikram Dmitri Sergei Pavel Mikhail Lars Sven Erik Anders
Henrik Mehmet Emre Kwame Kofi Tunde Chidi Rafael Mateus Joao Pedro Andres Javier Enrique Alejandro""".split()

LAST = """Kim Lee Park Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez Gonzalez Wilson
Anderson Moore Martin Thompson Harris Sanchez Clark Ramirez Lewis Robinson Young Allen Wright Scott Torres Nguyen Flores Adams
Baker Hall Rivera Campbell Mitchell Carter Roberts Gomez Phillips Evans Turner Diaz Cruz Edwards Collins Reyes Stewart Morris
Morales Murphy Cook Rogers Gutierrez Ortiz Peterson Kelly Howard Ramos Ward Richardson Watson Chavez Mendoza Ruiz Hughes Alvarez
Castillo Sanders Patel Myers Ross Foster Jimenez Powell Jenkins Perry Sullivan Coleman Butler Henderson Barnes Gonzales Fisher
Vasquez Simmons Romero Patterson Hamilton Graham Reynolds Griffin Moreno Hayes Bryant Herrera Gibson Ellis Tran Medina Aguilar
Stevens Ford Castro Owens Fernandez McDonald Washington Wells Vargas Chen Freeman Webb Tucker Guzman Burns Crawford Olson Simpson
Porter Mendez Silva Shaw Snyder Dixon Munoz Hicks Holmes Palmer Wagner Robertson Boyd Salazar Mills Meyer Schmidt Garza Ferguson
Nichols Stephens Soto Weaver Gardner Payne Dunn Kelley Hawkins Arnold Pierce Vazquez Hansen Peters Santos Hart Bradley Knight
Cunningham Duncan Armstrong Carroll Andrews Alvarado Delgado Berry Perkins Hoffman Johnston Matthews Pena Richards Contreras
Willis Carpenter Sandoval Guerrero Chapman Rios Estrada Ortega Watkins Greene Nunez Wheeler Valdez Burke Larson Maldonado Morrison
Carlson Dominguez Carr Lawson Jacobs Lynch Singh Vega Bishop Montgomery Jensen Harvey Williamson Gilbert Sims Espinoza Howell
Wong Reid Hanson McCoy Garrett Burton Fuller Wang Weber Welch Rojas Marquez Fields Yang Padilla Walsh Bowman Schultz Fowler
Mejia Davidson Acosta Brewer Holland Juarez Newman Pearson Curtis Cortez Schneider Barrett Navarro Figueroa Keller Avila Wade
Molina Stanley Hopkins Campos Barnett Bates Chambers Caldwell Beck Lambert Miranda Byrd Craig Ayala Lowe Frazier Powers Neal
Carrillo Sutton Fleming Rhodes Shelton Schwartz Norris Jennings Watts Duran Cohen McDaniel Moran Parks Steele Vaughn Becker Holt
Barker Hale Benson Haynes Horton Lyons Pham Graves Thornton Wolfe Cabrera McKinney Zimmerman Dawson Lara Fletcher McCarthy Robles
Cervantes Solis Erickson Reeves Chang Klein Salinas Fuentes Baldwin Velasquez Hardy Higgins Aguirre Lin Cummings Chandler Barber
Bowen Ochoa Robbins Liu Ramsey Griffith Blair Cardenas Pacheco Calderon Swanson Chan Rivas Khan Rodgers Serrano Fitzgerald
Rosales Stevenson Christensen Manning Gill Harmon McGee Doyle Garner Newton Burgess Walton Trujillo Adkins Brady Goodman Webster
Goodwin Fischer Huang Montoya Mullins Castaneda Malone Hubbard Hodges Zhang Guerra Valencia Saunders Franco Gallagher Hammond
Hampton Townsend Ingram Gallegos Barton Schroeder Camacho Strickland Colon Parsons Harrington Glover Osborne Buchanan Patton
Ibarra Suarez Bowers Orozco Salas Cobb Gibbs Andrade Bauer Conner Moody Escobar McGuire Lloyd Mueller Hartman Kramer McBride
Lindsey Velazquez Norton McCormick Flynn Yates Hogan Macias Villanueva Zamora Pratt Stokes Ballard Brock Villarreal Barrera
Pineda Burnett Mercado Santana Shepherd Bautista Shaffer Trevino McKenzie Olsen Cochran Morton Wilkins Petersen Briggs Shah Roth
Nicholson Holloway Lozano Rangel Hoover Arias Valenzuela Meyers Weiss Underwood Greer Whitaker Decker Yoder Collier Zuniga Wilcox
Melendez Roberson Larsen Conley Davenport Copeland Massey Huff Rocha Pittman Huynh Randall Singleton Combs Mathis Skinner Bradford
Galvan Boone Kirby Wilkinson Bridges Atkinson Velez Meza Hodge Villa Abbott Tapia Sosa Sweeney Farrell Barron Phelps Dickerson
Foley Atkins Bonilla Acevedo Benitez Zavala Hensley Cisneros Harrell Shields Rubio Choi Huffman Boyer Garrison Arroyo Kane
Hancock Callahan Dillon Cline Wiggins Grimes Arellano Melton Beltran Pitts Parrish Ponce Booth Koch Brennan McDowell Cantu Baxter
Hutchinson Kaur Berg Wiley Gilmore Russo Villegas Hobbs Wilkerson Ahmed Beard McClain Montes Rosario Henson Levy Mosley Petty
Herring Hester Brandt Fritz Okafor Mensah Nakamura Tanaka Suzuki Watanabe Yamamoto Kobayashi Sato Takahashi Ito Moreau Laurent
Dubois Lefebvre Fontaine Rossi Ricci Bianchi Romano Colombo Novak Kowalski Nowak Horvat Kovacs Petrov Ivanov Volkov Sokolov Popescu
Papadopoulos Yilmaz Demir Kaya Celik Haddad Nasser Karimi Rahimi Hosseini Ahmadi Mehta Sharma Gupta Iyer Reddy Nair Kapoor Malik
Qureshi Chowdhury Rahman Hussain Abubakar Adeyemi Eze Mwangi Otieno Banda Dlamini""".split()

CITIES = """Lisbon|Porto|Madrid|Barcelona|Valencia|Seville|Paris|Lyon|Marseille|Nice|Bordeaux|Toulouse|Berlin|Munich|Hamburg|Cologne|
Frankfurt|Stuttgart|Dresden|Leipzig|Vienna|Salzburg|Zurich|Geneva|Basel|Bern|Rome|Milan|Naples|Turin|Venice|Bologna|Genoa|Palermo|
Amsterdam|Rotterdam|Utrecht|Brussels|Antwerp|Ghent|Copenhagen|Aarhus|Stockholm|Gothenburg|Malmo|Oslo|Bergen|Helsinki|Tampere|
Dublin|Cork|Galway|London|Manchester|Liverpool|Leeds|Bristol|Glasgow|Edinburgh|Cardiff|Belfast|Prague|Brno|Warsaw|Krakow|Gdansk|
Budapest|Bucharest|Sofia City|Athens|Thessaloniki|Istanbul|Ankara|Izmir|Cairo|Alexandria|Casablanca|Marrakesh|Tunis|Nairobi|
Mombasa|Lagos|Accra|Dakar|Johannesburg|Cape Town|Durban|Addis Ababa|Kampala|Tokyo|Osaka|Kyoto|Nagoya|Sapporo|Fukuoka|Seoul|Busan|
Incheon|Beijing|Shanghai|Shenzhen|Guangzhou|Chengdu|Hangzhou|Taipei|Kaohsiung|Hong Kong|Singapore|Bangkok|Chiang Mai|Hanoi|
Manila|Cebu|Jakarta|Bandung|Surabaya|Kuala Lumpur|Penang|Delhi|Mumbai|Bangalore|Chennai|Kolkata|Hyderabad|Pune|Jaipur|Karachi|
Lahore|Dhaka|Kathmandu|Colombo|Dubai|Abu Dhabi|Doha|Muscat|Riyadh|Jeddah|Amman|Beirut|Tehran|Isfahan|Shiraz|Tabriz|Baku|Tbilisi|
Yerevan|Almaty|Tashkent|Sydney|Melbourne|Brisbane|Perth|Adelaide|Auckland|Wellington|Christchurch|Toronto|Montreal|Vancouver|
Calgary|Ottawa|Boston|Chicago|Denver|Seattle|Portland|Dallas|Houston|Miami|Atlanta|Phoenix|Detroit|Philadelphia|San Diego|
San Francisco|Los Angeles|Nashville|Minneapolis|Pittsburgh|Baltimore|Orlando|Tampa|New Orleans|Salt Lake City|Mexico City|
Guadalajara|Monterrey|Havana|Bogota|Medellin|Lima|Quito|Valparaiso|Buenos Aires|Cordoba|Montevideo|Sao Paulo|Rio de Janeiro|
Brasilia|Recife|Caracas|La Paz""".replace("\n", "").split("|")

COMPANY_PREFIX = """Northwind Bluepeak Silverline Ironwood Brightwater Redstone Clearview Greenfield Oakridge Stonebridge Highland Lakeside
Westbrook Eastgate Suncrest Moonrise Starfield Riverbend Pinecrest Maplewood Cedarline Goldleaf Copperfield Crystalview Deepwater
Falconer Harborview Horizon Keystone Lighthouse Meridian Nimbus Orbital Pinnacle Quantum Radiant Summitline Trident Vertex Zenith
Apex Beacon Cobaltix Deltaform Emberly Frontier Granite Helix Indigo Juniper Kestrel Lumen Mosaic Novaline Onyx Polaris Quartz
Ravenwood Sterling Tundra Umbra Vantage Willowby Xenon Zephyr Bramble Cascade Driftwood Evergreen Foxglove Glacier Hollowell Ivoryx
Kingfisher Larchmont Marbleton Nectarine Oakhurst Peregrine Quillon Redwood Saffron Tidewater Upland Vireo Wexford Yarrow""".split()
COMPANY_SUFFIX = """Labs Systems Logistics Health Analytics Robotics Media Foods Energy Capital Studios Software Design Motors Pharma
Consulting Insurance Airlines Textiles Biotech""".split()

JOBS = """nurse|architect|accountant|teacher|pharmacist|electrician|plumber|software engineer|data analyst|graphic designer|chef|
dentist|veterinarian|lawyer|paralegal|journalist|photographer|physiotherapist|pilot|flight attendant|librarian|firefighter|
police officer|civil engineer|mechanical engineer|product manager|project manager|marketing manager|sales representative|
real estate agent|financial advisor|investment banker|economist|statistician|biologist|chemist|physicist|geologist|surgeon|
radiologist|midwife|paramedic|carpenter|welder|baker|barista|tailor|florist|interior designer|translator|copywriter|editor|
illustrator|animator|game developer|web developer|systems administrator|network engineer|security analyst|UX researcher|
HR manager|recruiter|social worker|psychologist|optometrist|nutritionist|personal trainer|yoga instructor|music teacher|violinist|
pianist|sound engineer|film director|producer|actor|curator|archaeologist|historian|professor|lecturer|school principal|tutor|
consultant|auditor|tax advisor|insurance broker|logistics coordinator|warehouse manager|truck driver|bus driver|tour guide|
travel agent|hotel manager|event planner|wedding planner|landscape architect|urban planner|dietitian|zoologist|marine biologist|
astronomer|lab technician|dental hygienist|speech therapist|occupational therapist""".replace("\n", "").split("|")

STREETS = """Elm Oak Maple Pine Cedar Birch Willow Chestnut Walnut Spruce Aspen Poplar Laurel Magnolia Juniper Hawthorn Sycamore Cypress
Linden Alder Holly Beech Hickory Mulberry Palm Lake River Hill Valley Meadow Forest Garden Bridge Mill Church School Market Station
Harbor Ocean Shore Spring Sunset Sunrise Highland Lincoln Washington Jefferson Madison Franklin Adams Monroe Grant Kennedy Albert
Queen King Prince Castle Abbey Chapel Tower Kingsway Broadway Main High Front Center North South East West Union Liberty Commerce
Canal Railway Orchard Vine Lily Daisy Primrose Clover Heather Fern Thistle Acorn Bramble Copper Silver Granite Quarry Ridge Summit
Canyon Prairie Glen Brook Creek Falls Grove Hollow Heights Park""".split()
STREET_TYPES = "Street Avenue Road Lane Drive Court Place Boulevard Way Terrace".split()

PETS = """Biscuit Mochi Pickles Waffles Noodle Peanut Muffin Pumpkin Nugget Tofu Sushi Ziggy Bandit Rocket Shadow Smokey Ginger Coco
Oreo Snickers Twix Bubbles Sprinkles Buttons Whiskers Mittens Socks Boots Patches Pebbles Pippin Gizmo Gadget Widget Chewie Yoda
Frodo Bilbo Gandalf Simba Nala Mufasa Bambi Thumper Dumbo Scooby Snoopy Garfield Sylvester Tigger Pooh Paddington Rusty Dusty
Sandy Sunny Stormy Blizzard Comet Meteor Cosmo Astro Apollo Zeus Thor Loki Odin Hermes Pluto Juno Bruno Rex Tank Moose Hippo
Koala Papaya Basil Sage Marshmallow Cupcake Brownie Pretzel Churro Dumpling Wasabi Ramen Kimchi Falafel Taco Nacho Chorizo
Meatball Pesto Gnocchi Biscotti Cannoli Mango Kiwi""".split()

FOODS = """peanuts|walnuts|almonds|cashews|pistachios|hazelnuts|pecans|shellfish|shrimp|crab|lobster|oysters|mussels|salmon|tuna|
eggs|milk|cheese|wheat|gluten|soy|sesame|mustard|celery|lupin|kiwi fruit|strawberries|bananas|avocado|tomatoes|peaches|apples|
cherries|pineapple|coconut|chocolate|garlic|onions|mushrooms|corn|chickpeas|lentils|oats|rye|cinnamon|buckwheat|squid|clams|
scallops|anchovies|sardines|plums|apricots|melon|papaya fruit|eggplant|bell peppers|cilantro|fennel|poppy seeds|sunflower seeds|
pumpkin seeds|macadamia nuts|brazil nuts|pine nuts|honey|yogurt|cream|butter|barley|spelt|quinoa|chili|paprika|cumin|nutmeg|
vanilla|cocoa|coffee|tea|citrus|grapefruit|oranges|lemons|limes|blueberries|raspberries""".replace("\n", "").split("|")

REST_ADJ = """Golden Blue Red Green Silver Little Old Lucky Happy Rustic Royal Hidden Secret Wild Spicy Sweet Salty Smoky Crispy
Hungry Humble Urban Velvet Copper Iron Jade Ivory Crimson Amber Coral Dancing Sleepy Brave Clever Gentle Jolly Merry Noble Proud""".split()
REST_NOUN = """Spoon Fork Ladle Kettle Lotus Bamboo Olive Lemon Pepper Saffron Fig Plum Pear Truffle Dumpling Noodle Burrito Pizza
Bagel Bistro Kitchen Table Garden Lantern Dragon Tiger Rooster Goat Duck Oyster Anchor Mill Barrel Cellar Hearth Oven Grill Wok
Skillet Teapot Basket Platter Saucer Whisk Rolling Pin Cauldron Griddle Tandoor Samovar""".split()

WIFI_WORDS = """maple orbit river falcon velvet copper lantern meadow ember harbor quartz willow summit cobalt prism tundra
cactus breeze pixel rocket comet nimbus lotus marble cedar sparrow thistle jasper onyx zephyr glacier canyon voyage""".split()
AIRLINES = "LH BA AF KL UA DL AA IB AZ LX OS SK AY EK QR TK SQ CX JL NH QF AC LA AV".split()
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
MONTHS = "January February March April May June July August September October November December".split()

# ------------------------------------------------------------------------------------------------------------- attributes
# owner: person | self.  vtype: value generator.  confuser: attribute whose absence questions test similar-attribute near misses.
ATTRS = {
    "phone": dict(owner="person", canon="phone number", vtype="phone", confuser="work_phone",
                  alias_tr=["phone number", "phone", "cell", "cell number", "mobile"], alias_ood=["telephone number"],
                  s_tr=["You can reach {N} at {v}.", "Call {N} on {v}.", "{N} picks up calls at {v}."],
                  s_ood=["{N} answers the telephone at {v}."],
                  q_tr=["How can I reach {N} by phone?", "What number do I call to reach {N}?"],
                  q_val=["Which number should I dial for {N}?"],
                  q_ood=["If I wanted to ring {N}, which digits would I punch in?"],
                  upd=["{N} got a new cell number: {v}."]),
    "work_phone": dict(owner="person", canon="work phone", vtype="phone", confuser="phone",
                       alias_tr=["work phone", "office phone", "work number", "office number"], alias_ood=["business line"],
                       s_tr=["At work, {N} can be reached on {v}.", "{N}'s office line is {v}."],
                       s_ood=["During business hours {N} is reachable on {v}."],
                       q_tr=["How do I call {N} at work?", "What number reaches {N} at the office?"],
                       q_val=["Which number gets me {N} during work hours?"],
                       q_ood=["What is the business line I would use for {N}?"],
                       upd=[]),
    "email": dict(owner="person", canon="email address", vtype="email", confuser="work_email",
                  alias_tr=["email", "email address", "personal email"], alias_ood=["electronic mail address"],
                  s_tr=["You can email {N} at {v}.", "{N} reads messages sent to {v}."],
                  s_ood=["Electronic messages for {N} go to {v}."],
                  q_tr=["Where do I email {N}?", "What address do I use to email {N}?"],
                  q_val=["Where should I send {N} an email?"],
                  q_ood=["To which inbox should I write when contacting {N}?"],
                  upd=[]),
    "work_email": dict(owner="person", canon="work email", vtype="email", confuser="email",
                       alias_tr=["work email", "office email"], alias_ood=["corporate mailbox"],
                       s_tr=["At work, {N} uses the email {v}.", "{N}'s office account is {v}."],
                       s_ood=["{N}'s corporate account is {v}."],
                       q_tr=["What email does {N} use at work?", "Where do I email {N} about work?"],
                       q_val=["Which address reaches {N} at the office?"],
                       q_ood=["Which corporate mailbox belongs to {N}?"],
                       upd=[]),
    "birthday": dict(owner="person", canon="birthday", vtype="birthdate", confuser="anniversary",
                     alias_tr=["birthday", "date of birth", "birth date"], alias_ood=["natal day"],
                     s_tr=["{N} was born on {v}.", "{N} celebrates {pos} birthday on {v}."],
                     s_ood=["{N} came into the world on {v}."],
                     q_tr=["When was {N} born?", "When does {N} celebrate {pos} birthday?"],
                     q_val=["What day was {N} born?"],
                     q_ood=["When should I send {N} a card for turning a year older?"],
                     upd=[]),
    "anniversary": dict(owner="person", canon="wedding anniversary", vtype="weddate", confuser="birthday",
                        alias_tr=["anniversary", "wedding anniversary"], alias_ood=["date of the nuptials"],
                        s_tr=["{N} got married on {v}.", "{N}'s wedding was on {v}."],
                        s_ood=["{N} tied the knot on {v}."],
                        q_tr=["When did {N} get married?", "When was {N}'s wedding?"],
                        q_val=["What date did {N} marry?"],
                        q_ood=["When did {N} tie the knot?"],
                        upd=[]),
    "city": dict(owner="person", canon="home city", vtype="city", confuser=None,
                 alias_tr=["city", "home city", "town"], alias_ood=["municipality"],
                 s_tr=["{N} lives in {v}.", "{N} is based in {v}."],
                 s_ood=["{N} resides in {v}."],
                 q_tr=["Where does {N} live?", "Which city is {N} based in?"],
                 q_val=["What city does {N} call home?"],
                 q_ood=["In which city does {N} reside?"],
                 upd=["{N} moved to {v}.", "{N} relocated to {v}."]),
    "employer": dict(owner="person", canon="employer", vtype="company", confuser=None,
                     alias_tr=["employer", "company", "workplace"], alias_ood=["firm"],
                     s_tr=["{N} works at {v}.", "{N} has a job at {v}."],
                     s_ood=["{N} is on the payroll of {v}."],
                     q_tr=["Where does {N} work?", "Which company does {N} work for?"],
                     q_val=["Who does {N} work for?"],
                     q_ood=["Which firm employs {N}?"],
                     upd=["{N} now works at {v}.", "{N} switched jobs and joined {v}."]),
    "job": dict(owner="person", canon="job title", vtype="job", confuser=None,
                alias_tr=["job", "job title", "role", "position"], alias_ood=["occupation"],
                s_tr=["{N} works as {art} {v}.", "{N} has been {art} {v} for years."],
                s_ood=["{N} earns a living as {art} {v}."],
                q_tr=["What does {N} do for a living?", "What is {N}'s line of work?"],
                q_val=["What kind of work does {N} do?"],
                q_ood=["What profession does {N} practice?"],
                upd=["{N} was promoted to {v}.", "{N} is now working as {art} {v}."]),
    "address": dict(owner="person", canon="home address", vtype="address", confuser=None,
                    alias_tr=["address", "home address", "street address"], alias_ood=["domicile"],
                    s_tr=["{N}'s place is at {v}.", "{N} lives at {v}."],
                    s_ood=["{N} dwells at {v}."],
                    q_tr=["Where is {N}'s place?", "What street does {N} live on?"],
                    q_val=["Where is {N}'s house?"],
                    q_ood=["Where is {N}'s dwelling located?"],
                    upd=["{N} moved into a new place at {v}."]),
    "pet": dict(owner="person", canon="pet's name", vtype="pet", confuser=None,
                alias_tr=["pet's name", "dog's name", "pet"], alias_ood=["companion animal's name"],
                s_tr=["{N} has a dog named {v}.", "{N}'s dog is called {v}."],
                s_ood=["{N} adopted a puppy and named it {v}."],
                q_tr=["What is {N}'s dog called?", "What did {N} name {pos} dog?"],
                q_val=["What's the name of {N}'s dog?"],
                q_ood=["What does {N} call {pos} furry friend?"],
                upd=[]),
    "allergy": dict(owner="person", canon="allergy", vtype="food", confuser=None,
                    alias_tr=["allergy", "food allergy"], alias_ood=["intolerance"],
                    s_tr=["{N} is allergic to {v}.", "{N} can't eat {v}."],
                    s_ood=["{N} breaks out in hives after eating {v}."],
                    q_tr=["What is {N} allergic to?", "What can't {N} eat?"],
                    q_val=["What food should {N} avoid?"],
                    q_ood=["What ingredient must I leave out when preparing a meal for {N}?"],
                    upd=[]),
    "restaurant": dict(owner="person", canon="favorite restaurant", vtype="restaurant", confuser=None,
                       alias_tr=["favorite restaurant", "favourite restaurant"], alias_ood=["preferred eatery"],
                       s_tr=["{N} loves eating at {v}.", "{N}'s favorite place to eat is {v}."],
                       s_ood=["{N} always dines at {v}."],
                       q_tr=["Where does {N} like to eat?", "Where should I take {N} for dinner?"],
                       q_val=["Which restaurant does {N} love?"],
                       q_ood=["Which eatery would {N} pick for a night out?"],
                       upd=[]),
    "plate": dict(owner="person", canon="license plate", vtype="plate", confuser=None,
                  alias_tr=["license plate", "plate number", "car plate"], alias_ood=["registration tag"],
                  s_tr=["{N}'s car has the plate {v}.", "{N} drives a car with plate {v}."],
                  s_ood=["{N}'s vehicle is registered as {v}."],
                  q_tr=["What plate is on {N}'s car?", "What's the plate on {N}'s car?"],
                  q_val=["Which plate does {N}'s car have?"],
                  q_ood=["How is {N}'s vehicle registered?"],
                  upd=[]),
    "door_code": dict(owner="person", canon="door code", vtype="code", confuser=None,
                      alias_tr=["door code", "gate code", "entry code"], alias_ood=["keypad combination"],
                      s_tr=["To get into {N}'s building, enter {v}.", "{N}'s front door opens with {v}."],
                      s_ood=["{N}'s keypad accepts {v}."],
                      q_tr=["How do I get into {N}'s building?", "What do I enter at {N}'s door?"],
                      q_val=["What opens {N}'s front door?"],
                      q_ood=["Which combination unlocks {N}'s entrance?"],
                      upd=[]),
    "meeting": dict(owner="person", canon="next meeting", vtype="slot", confuser=None,
                    alias_tr=["next meeting", "meeting time"], alias_ood=["upcoming rendezvous"],
                    s_tr=["I'm meeting {N} on {v}.", "My next meeting with {N} is on {v}."],
                    s_ood=["{N} and I are getting together on {v}."],
                    q_tr=["When am I meeting {N}?", "When is my next meeting with {N}?"],
                    q_val=["When do I see {N} next?"],
                    q_ood=["When is my upcoming rendezvous with {N}?"],
                    upd=["The meeting with {N} moved to {v}.", "I rescheduled {N} to {v}."]),
    # ---- self entity ('my ...')
    "wifi": dict(owner="self", canon="Wi-Fi password", vtype="wifi", confuser=None,
                 alias_tr=["Wi-Fi password", "wifi password", "network password"], alias_ood=["router passphrase"],
                 s_tr=["The home network password is {v}.", "To get online at home, use {v}."],
                 s_ood=["Guests connect to my router with {v}."],
                 q_tr=["How do I log into my home network?", "What's the password for the home internet?"],
                 q_val=["What do I type to get online at home?"],
                 q_ood=["What should guests enter to connect to my router?"],
                 upd=[]),
    "parking": dict(owner="self", canon="parking spot", vtype="parking", confuser=None,
                    alias_tr=["parking spot", "parking space"], alias_ood=["parking bay"],
                    s_tr=["I parked in {v}.", "My car is in spot {v}."],
                    s_ood=["I left the car at bay {v}."],
                    q_tr=["Where did I park?", "Where is my car parked?"],
                    q_val=["Which spot is my car in?"],
                    q_ood=["At which bay did I leave the car?"],
                    upd=["I moved my car to {v}."]),
    "passport": dict(owner="self", canon="passport number", vtype="passport", confuser=None,
                     alias_tr=["passport number", "passport"], alias_ood=["travel document number"],
                     s_tr=["My passport reads {v}.", "The number on my passport is {v}."],
                     s_ood=["My travel document is numbered {v}."],
                     q_tr=["What number is on my passport?", "What's printed on my passport?"],
                     q_val=["What does my passport say?"],
                     q_ood=["What is the identifier on my travel document?"],
                     upd=[]),
    "flight": dict(owner="self", canon="flight number", vtype="flight", confuser=None,
                   alias_tr=["flight number", "flight"], alias_ood=["departure code"],
                   s_tr=["I'm flying on {v}.", "My flight is {v}."],
                   s_ood=["I'm booked to depart on {v}."],
                   q_tr=["Which flight am I taking?", "What flight am I on?"],
                   q_val=["What flight did I book?"],
                   q_ood=["Which plane am I boarding?"],
                   upd=["My flight changed to {v}."]),
    "dentist": dict(owner="self", canon="dentist appointment", vtype="slot", confuser=None,
                    alias_tr=["dentist appointment", "dental appointment"], alias_ood=["checkup with the dentist"],
                    s_tr=["I'm seeing the dentist on {v}.", "My dentist booked me for {v}."],
                    s_ood=["My teeth get cleaned on {v}."],
                    q_tr=["When do I see the dentist?", "When is my dentist visit?"],
                    q_val=["When am I going to the dentist?"],
                    q_ood=["When are my teeth getting cleaned?"],
                    upd=["The dentist moved my appointment to {v}."]),
    "locker": dict(owner="self", canon="locker code", vtype="code", confuser=None,
                   alias_tr=["locker code", "gym locker code"], alias_ood=["padlock setting"],
                   s_tr=["My gym locker opens with {v}.", "The code for my locker is {v}."],
                   s_ood=["My padlock at the gym is set to {v}."],
                   q_tr=["How do I open my gym locker?", "What opens my locker?"],
                   q_val=["What's the code to my locker?"],
                   q_ood=["What releases my padlock at the gym?"],
                   upd=[]),
}

# Words that must never occur in train/val text (novel words of OOD aliases and OOD templates; checked at build time).
OOD_NOVEL_WORDS = """telephone ring digits punch business reachable electronic inbox contacting corporate mailbox natal world
turning older nuptials tied knot municipality resides reside payroll firm employs occupation earns profession practice domicile
dwells dwelling companion animal adopted puppy furry intolerance hives ingredient preparing meal eatery dines preferred registration
tag vehicle registered keypad combination unlocks entrance rendezvous upcoming getting together router passphrase guests connect
bay identifier document numbered depart departure plane boarding checkup teeth cleaned padlock setting releases dig belongs
pull jot scratch swapped heads wipe dropping surprising""".split()

# ------------------------------------------------------------------------------------------------------- generic templates
GEN_S = {"train": ["{N}'s {alias} is {v}.", "The {alias} for {N} is {v}.", "Note: {N}'s {alias} is {v}.", "{N}, {alias}: {v}.",
                   "Just so I remember, {N}'s {alias} is {v}."],
         "ood": ["Jot this down: {v} is {N}'s {alias}.", "For the record, {N} has {v} as {pos} {alias}."]}
GEN_S_SELF = {"train": ["{M} {alias} is {v}.", "Note to self: {m} {alias} is {v}.", "Reminder: {m} {alias} is {v}."],
              "ood": ["Jot this down: {v} is {m} {alias}.", "For the record, I have {v} as {m} {alias}."]}
GEN_Q = {"train": ["What is {N}'s {alias}?", "What's {N}'s {alias}?", "Tell me {N}'s {alias}.", "Do you have {N}'s {alias}?",
                   "Remind me of {N}'s {alias}.", "{N}'s {alias}?"],
         "val": ["Can you find {N}'s {alias}?", "I need {N}'s {alias}."],
         "ood": ["Could you dig up the {alias} that belongs to {N}?", "Pull up the {alias} on file for {N}."]}
GEN_Q_SELF = {"train": ["What is {m} {alias}?", "What's {m} {alias}?", "Remind me of {m} {alias}.", "Do you have {m} {alias}?"],
              "val": ["Can you find {m} {alias}?"],
              "ood": ["Could you dig up the {alias} that belongs to me?", "Pull up the {alias} on file for me."]}
PRON_Q = {"train": ["What is {pos} {alias}?", "What's {pos} {alias}?", "Do you have {pos} {alias}?", "Remind me of {pos} {alias}."],
          "val": ["Can you find {pos} {alias}?"],
          "ood": ["Could you dig up the {alias} that belongs to {obj}?", "Pull up the {alias} on file for {obj}."]}
CTX_TURN = {"train": ["I'm having lunch with {N} tomorrow.", "{N} just texted me.", "I need to send something to {N}.",
                      "I'm about to call {N}."],
            "val": ["I'm visiting {N} next week."],
            "ood": ["{N} is dropping by this afternoon.", "Thinking of surprising {N} soon."]}
CTX_TURN2 = {"train": ["I'm having dinner with {N1} and {N2} tonight.", "{N1} introduced me to {N2} today."],
             "val": ["{N1} and {N2} are coming over later."],
             "ood": ["{N1} is dropping by with {N2} this afternoon."]}
UPD_GEN = {"train": ["{N} changed {pos} {alias} to {v}.", "Update: {N}'s {alias} is now {v}.", "{N}'s new {alias} is {v}."],
           "ood": ["Heads up, {N} swapped {pos} {alias} for {v}."]}
UPD_SELF = {"train": ["{M} new {alias} is {v}.", "I changed {m} {alias} to {v}."], "ood": ["Heads up, I swapped {m} {alias} for {v}."]}
CORR_GEN = {"train": ["Correction: {N}'s {alias} is {v}, not {old}."], "ood": ["Scratch that: {N}'s {alias} is actually {v}, not {old}."]}
CORR_SELF = {"train": ["Correction: {m} {alias} is {v}, not {old}."], "ood": ["Scratch that: {m} {alias} is actually {v}, not {old}."]}
DEL_GEN = {"train": ["Please forget {N}'s {alias}.", "Delete {N}'s {alias} from my notes.", "{N}'s {alias} is no longer valid."],
           "ood": ["Wipe out the {alias} I saved for {N}."]}
DEL_SELF = {"train": ["Please forget {m} {alias}.", "Delete {m} {alias} from my notes."], "ood": ["Wipe out the {alias} I saved for myself."]}
COREF_INTRO = {"train": ["I ran into {N} at the station.", "I had coffee with {N} this morning.", "{N} called me yesterday."],
               "ood": ["{N} stopped me in the hallway today."]}
COREF_FACT = {"train": ["{Sub} said {pos} {alias} is {v}.", "{Sub} told me {pos} {alias} is {v}."],
              "ood": ["{Sub} mentioned in passing that {pos} {alias} is {v}."]}
COREF_INTRO2 = {"train": ["I met {N1} and {N2} at the conference.", "{N1} came to the party with {N2}."],
                "ood": ["{N1} stopped by the office with {N2}."]}
MULTI_SAME = {"train": ["{N1}'s {alias} is {v1}, and {N2}'s is {v2}."], "ood": ["{N1} has {v1} as {pos1} {alias}, whereas {N2} has {v2}."]}
MULTI_ATTR = {"train": ["{N}'s {alias1} is {v1} and {pos} {alias2} is {v2}."], "ood": ["{N} has {v1} as {pos} {alias1}, plus {v2} as {pos} {alias2}."]}
LIST_CARD = {"train": ["Contact card for {N}: {items}."], "ood": ["Details on {N} follow: {items}."]}

DISTRACT_PLAIN = """The weather was lovely today.|I need to buy groceries after work.|The train was late again this morning.|
Remember to water the plants.|The new episode of that show was great.|I should go to bed earlier.|Traffic on the bridge was terrible.|
The coffee machine at work broke again.|I finished reading the novel last night.|The park was crowded on Sunday.|
I need to renew my library card.|The neighbors are renovating their kitchen.|It rained for most of the afternoon.|
The concert tickets sold out quickly.|I want to try the new bakery downtown.|My laptop battery drains too fast.|
The gym was almost empty tonight.|I forgot my umbrella at the office.|The movie was longer than expected.|
I should call the bank about the fees.|The garden needs more sunlight.|Dinner was pasta with roasted vegetables.|
The power went out for an hour.|I signed up for a pottery class.|The elevator in my building is out of order.""".replace("\n", "").split("|")
DISTRACT_ENT = ["I saw {N} at the gym today.", "{N} recommended a great book.", "{N} said hello at the party.", "I owe {N} a coffee.",
                "{N} is training for a marathon.", "{N} sent me a funny video."]
DISTRACT_ASK = ["I should ask {N} for {pos} {alias}.", "I forgot to save {N}'s {alias}.", "Need to get {N}'s {alias} at some point."]
DISTRACT_VAL = ["I saw {v} written on a napkin.", "Someone left {v} on a sticky note.", "The radio mentioned {v} twice."]
