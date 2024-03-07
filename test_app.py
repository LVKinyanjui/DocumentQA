import unittest
import gradio as gr
from app_ import app

class TestApp(unittest.TestCase):

    def test_app_launch(self):
        self.assertIsNotNone(app)
        app.close()
    
    def test_app_has_file_input(self):
        self.assertEqual(len(app.blocks), 1)
        self.assertIsInstance(app.blocks[0], gr.File)

if __name__ == '__main__':
    unittest.main()
import unittest
from app_ import read_documents

class TestReadDocuments(unittest.TestCase):

    def test_read_documents_valid(self):
        # Arrange
        filepath = 'valid.pdf'
        
        # Act
        documents = read_documents(filepath)
        
        # Assert
        self.assertIsInstance(documents, list)
        self.assertGreater(len(documents), 0)

    def test_read_documents_invalid(self):
        # Arrange
        filepath = 'invalid.pdf'
        
        # Act and Assert
        with self.assertRaises(OSError):
            read_documents(filepath)

    def test_read_documents_empty(self):
        # Arrange
        filepath = 'empty.pdf'
        
        # Act
        documents = read_documents(filepath)
        
        # Assert
        self.assertEqual(len(documents), 0)

