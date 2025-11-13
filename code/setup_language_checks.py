LANGUAGE_ENG = False

# Ausgaben

if LANGUAGE_ENG:
    avg_rating_dataset = "\nThe average rating of the dataset is:"
    matrix_properties = "The matrix has the following properties:\n"
    n0_entries = "Non-zero entries in the rating matrix:"
    movie_count = "\nNumber of movies in the rating matrix, i.e., number of columns:"
    user_count = "\nNumber of users in the rating matrix, i.e., number of rows:"
    columns = {
        "title": "Movie Title",
        "genres": "Genres",
        "releaseyear": "Release Year",
        "idx": "ID",
        "count": "Number of Ratings",
        "avg": "Average Rating",
        "std": "Standard Deviation",
    }
    suc_load = "The Netflix dataset has been successfully loaded.\n"
    labels = {"avg": "Average Rating",
                "std": "Standard Deviation of Rating",
                "count": "Number of Ratings",
                "releaseyear": "Release Year"}
    suc_load = "The Netflix dataset has been loaded successfully.\n"
    rating_histogram = "Rating Histogramm"
    count_rating = 'Number of Ratings'
    best_film = "What is the most popular film in the dataset?"
    oldes_newest_film = "What year are the oldest films in the dataset from? What year are the youngest films from?"
    rating_distribution = "How has the distribution of ratings changed over the years of release?"
    most_ratings = "Which film has been rated the most?"
    lowest_std = "For which film do the ratings of individual users deviate the least from the average rating?"
    your_ideas = "Your Ideas"
    context = "Context"
    procedure_and_recommendations = "Procedure and Recommendations"
    known_percentage = "Percentage of Known Films"


else:
    avg_rating_dataset = "\nDas durchschnittliche Rating des Datensatzes lautet:"
    matrix_propertys = "Die Matrix hat folgende Eigenschaften:\n"
    n0_entrys = "Nicht-Null Einträge in der Ratingmatrix:"
    movie_count = "\nAnzahl Filme in der Ratingmatrix, d.h. Anzahl der Spalten:"
    user_count = "\nAnzahl User in der Ratingmatrix, d.h. Anzahl der Zeilen:"
    columns = {
        "title": "Filmtitel",
        "genres": "Genres",
        "releaseyear": "Erscheinungsjahr",
        "idx": "ID",
        "count": "Anzahl Ratings",
        "avg": "durchschnittl. Rating",
        "std": "Standardabweichung",
    }
    suc_load = "Der Netflixdatensatz wurde erfolgreich geladen.\n"
    lables = {"avg": "durchschnittl. Rating",
                "std": "Standardabw. Rating",
                "count": "Anzahl Ratings",
                "releaseyear": "Erscheinungsjahr"}
    suc_load = "Der Netflixdatensatz wurde erfolgreich geladen.\n"
    rating_histogram = "Histogramm der Ratings"
    count_rating = 'Anzahl Ratings'
    best_film = "1. Beliebtester Film"
    oldes_newest_film = "2. Ältester und jüngste Film"
    rating_distribution = "3. Verteilung der Bewertungen"
    most_ratings = "4. Am häufigsten bewertet"
    lowest_std = "5. Geringste Abweichung von mittlerer Bewertung"
    your_ideas = "Deine Ideen"
    contex = "Zusammenhänge"
    procedure_and_recomendations = "Vorgehen und Vorschläge"
    known_percentage = "Prozent der bekannten Filme"
