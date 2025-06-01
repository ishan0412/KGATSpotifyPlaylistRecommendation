# Knowledge-Graph Neural Network for Spotify Playlist Recommendation
This project will implement and employ the Knowledge Graph Attention Network model described by [Wang *et al.* (2019)](https://dl.acm.org/doi/10.1145/3292500.3330989) to recommend songs for Spotify playlists. The model will be trained on over 2.5 million entries of existing playlists and their songs, obtained from the [Spotify Playlists](https://www.kaggle.com/datasets/andrewmvd/spotify-playlists) and [Spotify 1.2M+ Songs](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs) datasets on Kaggle. By taking in a graph representation of the connections between playlists, songs, artists, and albums, the model, learning patterns among them, will be able to intelligently suggest the most suitable songs to add to any given playlist.
## Project Structure
- `models`: References for the datasets' fields and the graph of playlists, songs, artists, and albums constructed from them
- `utils`: Functions for downloading and processing the datasets
- `main.py`: Driver code for the currently implemented functionality 
## Usage
The project steps completed thus far -- downloading the datasets and turning them into a graph -- are shown in `main.py`.
