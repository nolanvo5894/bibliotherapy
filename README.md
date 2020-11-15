# iyas.ai
There is one secret that every book lover knows: reading is healing. Being fully immersed in a good story, or pondering a poem's verse, brings about a sense of calmness and tranquility that cannot be found any other way. iyas.ai (from the Japanese word 'iyasu', which means to calm) is a recommender system for books and stories that could help lift up your mood whenever you are facing stress or unpleasant situations. Just type in how you are feeling and we will suggest a few books or stories that might just brighten up your day.<br>

The datasets used to build iyas.ai come from 2 sources:

* The Novel Cure* by Ella Berthoud and Susan Elderkin: this is where the inspiration for this project came from. Ella and Susan have created a marvelous list of books for you to read through your difficult times. However, there are only 751 books mentioned in The Novel Cure. I believe no literary experts could have read through the entirety of human literature, and so some machine learning 'magic' could step in and help guide you through your search for the mood-perfect books.

* * Goodreads: Goodreads is one the most active social networks for book lovers on the web. I scraped 50000 books from Goodreads' 'Best Books Ever' list with their titles, authors, and summary. Through similarity search using neural word embeddings, iyasai find the stories that match most closely with the mood you enter into the system and suggest them to you. Voilà!

Enjoy :D


