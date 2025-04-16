# Cluster YouTube Comments

A series of stdout-oriented Python scripts for processing, analyzing, and clustering YouTube comments using AI and FAISS.

## Overview

This set of scripts:

- Creates embeddings from YouTube comments (see below)
- Visualizes vector embeddings using dimensionality reduction techniques
- Clusters the reviews and provides description/category for each cluster

**Note:** This project uses DeepSeek to generate cluster descriptions. You will need a valid API key to access their service — see the .env setup in the Setup section below.

## Data Source

The project uses YouTube comment data from the [Peterson-Newman Interview: 50K YT Comments](https://www.kaggle.com/datasets/kanchana1990/peterson-newman-interview-50k-yt-comments) on Kaggle.

## Setup

1. Clone the repository:
```sh
git clone https://github.com/alexpederneschi/cluster_youtube_comments.git
cd cluster_youtube_comments
  ```
2. Start the application:
```sh
bash ./start.sh
```

3. Create a `.env` file with your DeepSeek API key:
```
API_KEY=your_api_key_here
  ```

## Download Instructions

1. Visit the [dataset page](https://www.kaggle.com/datasets/kanchana1990/peterson-newman-interview-50k-yt-comments)
2. Click the "Download" button (requires Kaggle account)
3. Extract the downloaded zip file
4. Place the CSV file in the `data/` directory:
```bash
mkdir -p data
mv jp_vs_cn.csv.csv data/comments.csv
```

## Usage

The scripts by default read from stdin and write to stdout. You can also specify input/output files on the command line (see help for each program).

Full pipeline example:
```bash
python src/tojson.py --input data/comments.csv --output data/comments.json
python src/preprocess.py --input data/comments.json --output data/comments.sample.json --emoji convert --sample 10000
python src/embed.py --input data/comments.sample.json --output data/comments.embed.json
python src/visualize.py --input data/comments.embed.json --output_dir output
python src/cluster.py --input data/comments.embed.json --output output/clusters.json 
```

### Scripts
  
- src/tojson.py

  Converts csv to line-oriented json file
- src/preprocess.py

  Handles emojis and sampling
- src/embed.py

  Embeds summary text
- src/visualize.py

  Visualizes vector embeddings using dimensionality reduction techniques
- src/cluster.py

  Clusters reviews and provides descriptions/categories of each cluster

## Requirements

- Docker and Docker Compose installed on your system 

## License

MIT License