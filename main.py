import indexer
import os


#scan through all files in /files

for file in os.listdir("files"):
    #check if the file is a pdf
    if file.endswith(".pdf"):
        #get the file path
        file_path = os.path.join("files", file)
        #index the file
        indexer.get_embeddings(file_path, force_regenerate=True)
        print(file)

print("-------------------------------")

query = input("What would you like to query? ")
print("-------------------------------")


# query now
print("Query: ", query)
print("-------------------------------")

segments = indexer.get_relevant_segments(query)


segmentsSorted = sorted(segments.items(), key=lambda x: x[1]['similarity'], reverse=True)

# calculate the average similarity for a given file and all its segments
file_similarities = {}
for segment, data in segmentsSorted:
    file_name = data['file_name']
    similarity = data['similarity']
    if file_name not in file_similarities:
        file_similarities[file_name] = {'score': 0, 'similarity': 0, 'count': 0}

    file_similarities[file_name]['similarity'] += similarity
    file_similarities[file_name]['count'] += 1

    if similarity > 0.2:
        file_similarities[file_name]['score'] += 1

# calculate the average similarity
for file_name, data in file_similarities.items():
    if data['count'] > 0:
        data['similarity'] /= data['count']
    else:
        data['similarity'] = 0


ranked_files = sorted(file_similarities.items(), key=lambda x: x[1]['similarity'], reverse=True)

#print the top 5 files
print("Top 5 Files: ")
for file_name, data in ranked_files[:5]:
    print(f"File: {file_name}, Average Similarity: {data['similarity']:.4f} Score: {data['score']:.4f}, Segments: {data['count']}")



print("\nTop 5 Segments:")
for segment, data in segmentsSorted[:5]:
    print(f"Segment: {segment}, Similarity: {data['similarity']:.4f}, File: {data['file_name']}")