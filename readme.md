Bugs:
There are still minor bugs left like when you select the 1st doc and ask for answers it gives the answer, now if you select another 2nd document and ask it ll give but again when I select the 1st document its not storing the vector database properly
I know why its not storing: Its because the vector db doesn't get reinitialized because of the frontend streamlit file not allowing for the same file to run the create_vector_store function again.
