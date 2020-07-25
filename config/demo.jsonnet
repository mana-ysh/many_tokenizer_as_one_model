{
  "dataset_reader": {
    "type": "multi_criterion_dataset"
  },
  "train_data_path": "./data/sample/train_dataset.json",
  "model": {
    "type": "multi_criterion_tokenizer",
    "char_embeddings": {
      "type": "basic",
      "token_embedders": {
          "tokens": {
              "num_embeddings": 100,
              "embedding_dim": 6 
          },
      }
    },
    "criterion_embeddings": {
      "num_embeddings": 7,
      "embedding_dim": 6
    },
    "encoder": {
      "type": "lstm",
      "input_size": 6,
      "hidden_size": 6,
    }
  },
  "data_loader": {
    "batch_size": 8
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 0.0001
    },
    "num_epochs": 5,
  }
}