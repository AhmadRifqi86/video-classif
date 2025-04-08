import pymongo
import os
import json
import csv
import pandas as pd


## Ini nanti dicoba di rumah, jumat sore atau malam
class MongoDBExporter:
    def __init__(self, 
                 uri="mongodb://localhost:27017/", 
                 database_name="video_classification", 
                 collection_name="classification_results"):
        """
        Initialize MongoDB connection
        
        :param uri: MongoDB connection URI
        :param database_name: Name of the database
        :param collection_name: Name of the collection
        """
        try:
            self.client = pymongo.MongoClient(uri)
            self.db = self.client[database_name]
            self.collection = self.db[collection_name]
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            raise

    def export_to_json(self, output_path='mongodb_export.json'):
        """
        Export all documents to a JSON file
        
        :param output_path: Path to save the JSON file
        :return: Path to the exported file
        """
        try:
            # Retrieve all documents
            documents = list(self.collection.find({}, {'_id': False}))
            
            # Write to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=4, ensure_ascii=False)
            
            print(f"Successfully exported {len(documents)} documents to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            raise

    def export_to_csv(self, output_path='mongodb_export.csv'):
        """
        Export all documents to a CSV file
        
        :param output_path: Path to save the CSV file
        :return: Path to the exported file
        """
        try:
            # Retrieve all documents
            documents = list(self.collection.find({}, {'_id': False}))
            
            if not documents:
                print("No documents found to export.")
                return None

            # Use pandas to convert documents to DataFrame and export
            df = pd.DataFrame(documents)
            df.to_csv(output_path, index=False)
            
            print(f"Successfully exported {len(documents)} documents to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            raise

    def export_to_excel(self, output_path='mongodb_export.xlsx'):
        """
        Export all documents to an Excel file
        
        :param output_path: Path to save the Excel file
        :return: Path to the exported file
        """
        try:
            # Retrieve all documents
            documents = list(self.collection.find({}, {'_id': False}))
            
            if not documents:
                print("No documents found to export.")
                return None

            # Use pandas to convert documents to DataFrame and export
            df = pd.DataFrame(documents)
            df.to_excel(output_path, index=False)
            
            print(f"Successfully exported {len(documents)} documents to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            raise
    def calculate_latency_statistics(self):
        """
        Calculate mean and standard deviation of latency values
        :return: Dictionary containing mean and standard deviation
        """
        try:
            # Retrieve all documents and extract latency values
            documents = list(self.collection.find({}, {'latency': 1, '_id': 0}))
            
            if not documents:
                print("No documents found to analyze.")
                return None
            
            # Extract latency values from documents
            latencies = [doc['latency'] for doc in documents if 'latency' in doc]
            
            if not latencies:
                print("No latency data found in documents.")
                return None
            
            # Convert to pandas Series for easy calculation
            latency_series = pd.Series(latencies)
            
            # Calculate statistics
            total_latency = latency_series.sum()
            mean_latency = latency_series.mean()
            std_dev_latency = latency_series.std()
            
            # Create results dictionary
            stats = {
                "count": len(latencies),
                "sum": total_latency,
                "mean_latency": mean_latency,
                "std_dev_latency": std_dev_latency,
                "min_latency": latency_series.min(),
                "max_latency": latency_series.max(),
                "median_latency": latency_series.median()
            }
            
            print(f"Latency Statistics:")
            print(f"Count: {stats['count']}")
            print(f"Sum: {stats['sum']}")
            print(f"Mean: {stats['mean_latency']:.4f}")
            print(f"Std Dev: {stats['std_dev_latency']:.4f}")
            print(f"Range: {stats['min_latency']:.4f} - {stats['max_latency']:.4f}")
            
            return stats
        
        except Exception as e:
            print(f"Error calculating latency statistics: {e}")
            raise

    def close_connection(self):
        """
        Close the MongoDB connection
        """
        self.client.close()


def main():
    # Example usage
    try:
        # Initialize exporter
        exporter = MongoDBExporter()
        
        # Export to different formats
        json_file = exporter.export_to_json()
        # csv_file = exporter.export_to_csv()
        # excel_file = exporter.export_to_excel()
        
        # Calculate and display latency statistics
        latency_stats = exporter.calculate_latency_statistics()
        
        # Close MongoDB connection
        exporter.close_connection()
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()