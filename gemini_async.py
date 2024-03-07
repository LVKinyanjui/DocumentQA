import aiohttp
import asyncio
import os, time

api_key = os.environ['GOOGLE_API_KEY']

async def get_http_response(session, url, sentence):
    headers = {
        'Content-Type': 'application/json'
        }
    
    data = {
        'model':'models/embedding-001',
        'content': {
            'parts': [{
                'text': sentence
            }]
            }
        }

    
    async with session.post(url, 
                    headers=headers,
                    json=data,
                    params={'key': api_key}) as resp:
        
        return await resp.json()       # Ensure you await, otherwise it raises
                                        # RuntimeWarning: coroutine 'ClientResponse.text' was never awaited


async def async_embed(texts: list[str]):
    """
    Process the texts and return the embeddings.
    Args:
        texts: The texts to be processed.
    Returns:
        embeddings: The embeddings of the texts.
    """

    url = 'https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent'

    async with aiohttp.ClientSession() as session:
        tasks = []
        for sentence in texts:
            tasks.append(asyncio.ensure_future(get_http_response(session, url, sentence)))

        embeddings = await asyncio.gather(*tasks)

        return embeddings

def embed(texts):
    """A wrapper for the asynchronous operation. Runs the coroutines and returns actual embeddings"""
    return asyncio.run(async_embed(texts))


if __name__ == '__main__':
    start_time = time.time()

    sentences = ['Hello World', 'This is a sentence', 'This is another sentence']
    results = asyncio.run(async_embed(sentences))
    print(results)

    print("--- %s seconds ---" % (time.time() - start_time))

   

    