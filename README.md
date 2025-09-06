# Textbook Question Answering (TQA)
The TQA dataset consists of 1,076 lessons from Life Science, Earth Science and Physical Science textbooks downloaded from www.ck12.org. Each lesson has a set of multiple choice questions that address concepts taught in that lesson. The number of choices varies from two to seven. TQA has a total of 26,260 questions including 12,567 having an accompanying diagram. TQA is split into a training, validation and test set at lesson level.The training set consists of 666 lessons and 15,154 questions, the validation set consists of 200 lessons and 5,309 questions and the test set consists of 210 lessons and 5,797 questions. On occasions, multiple lessons have an overlap in the concepts they teach. Care has been taken to group these lessons before splitting the data, so as to minimize the concept overlap between data splits.

## Source
This dataset was constructed in large part from material freely available as part of ck12's open-source
science curriculum. It is distributed under a Creative Commons Attribution – Non-Commerical 3.0 license.  

## Instructional Material
Lessons in the dataset each have limited amount of multimodal material that contains the
information needed to answer that lesson's text and diagram questions. Textual material is
broken down by subtopic within the lesson. Figures and instructional diagrams are linked
from the topic in which they appear.

## Questions
The dataset is split into a training, validation and test set at lesson level. The training set consists of 666 lessons and 15,154 questions,
the validation set consists of 200 lessons and 5,309 questions and the test set consists of 210 lessons and 5,797 questions.
On occasions, multiple lessons have an overlap in the concepts they teach.
Care has been taken to group these lessons before splitting the data, so as to minimize the concept overlap between splits.

Questions are separated by type in the dataset (nested under diagramQuestions and nonDiagramQuestions),
and are also distinguished by global id prefixes indicating an associated diagram (DQ_* and NDQ_*).

## Images
Images in the dataset are divided into four types:

question_images (diagrams referred to in diagram questions)

abc_question_images (question diagrams generated with letter labels)

teaching_images (diagrams which have detailed descriptions)

textbook_images (images from the instructional material)

Images are referenced in the dataset as relative paths to one of these directories.

## Dataset Structure
Here is the structure of the dataset json files for the train, test, and validation sets:

	[  ### at the top level is a list of "lessons" that are the atomic unit of the dataset
        { ### within a lesson are several fields, including questions and their related instructional material

        		"globalID": "global id string",
        		"lessonName": "name of lesson",
        		"questions": {
                "nonDiagramQuestions": {
                    "question global ID": {
                        "answerChoices": {
                            "a": {
                                "idStructural": "a.",
                                "processedText": "cleaned answer option string",
                                "rawText": "unprocessed answer option given in question"
                            },
                            .
                            .
                            ... additional answer options                            
                        },
                        "beingAsked": {
                            "processedText": "cleaned question string",
                            "rawText": "unprocessed question being asked"
                        },
                        "correctAnswer": {
                            "processedText": "cleaned correct answer string",
                            "rawText": "correct answer to question"
                        },
                        "globalID": "global id number",
                        "idStructural": "number identifier from source material",
                        "questionType": "this distinguishes MC and DA questions"
												"questionSubType": "detailed question category",
                 		}
                 		.
                 		.
                 		.... additional questions
                }
                "diagramQuestions": {
                			The same fields as non-diagram questions, plus:
                			"imagePath": "relative local path to image file"
                			 "imageName": "image file name"
                },
                "instructionalDiagrams": {
                		"global_id": {
                 		     "global id": "global id number",
                		     "imagePath": "relative local path to image file",
        					"imageName": "image name",
			    		    "rawText": "description of diagram",
			        		"processedText": "processed description of diagram"
        				},
        				...
                },
								"diagramAnnotations": {
									"diagram_name": [
									{
										"text": "the ground truth content of the text box",
										"rectangle": "the location of the text "
									},
									...
								},
                "topics": {
                		"Global Topic ID": {
                			"global ID": global id string,
                   		 "content": {
                        		"figures": [
                         		{
                             		   "caption": "figure caption text",
                                		"imagePath": "relative local path to image file"
                           		 }
								],
	                        "text": "Paragraph sized text explaining topic"
    		                		},
        		            	"orderID": "order of appearance within lesson",
      		            		"mediaLinks": {[
	        		            	list of youtube/ other video links extracted from the text]
                                            }
                                      }
                              }
                "adjunctTopics": {
                		"Vocabulary": {
                			dictionary of lesson vocabulary words and their definitions.
                		},
                		"section name": {
                		     "content": {
                        		"figures": [
                         		{
                             		   "caption": "figure caption text",
                                		"imagePath": "relative local path to image file"
                           		 }
								],
	                        "text": "Paragraph sized text explaining topic"
    		                		},
        		            		"orderID": "order of appearance within lesson",
        		            		"mediaLinks": {[
	        		            list of youtube/ other video links extracted from the text]
                                            }
                                      }
                              }
                		}

                }
        	  }                
        .
        .
				... repeated for additional lessons
       ]



* Top-Level Lesson: This is the atomic unit of the dataset, and the level at which questions are paired to instructional 	material, e.g "History of Life on Earth". The important components of a lesson are listed below.

	* questions: questions that should be answerable given the contents of the lesson.
		* nonDiagramQuestions: questions with no accompanying diagram needed to answer
		* diagramQuestions: questions with an accompanying diagram needed to answer
			* questionType: format of the question, multiple choice, fill-in-the-blank, true/false, diagram multiple choice
			* beingAsked: text of the question being posed
			* idStructural: the original number or letter identifier in the source, e.g. "1.", "2.", "A."
			* globalID: dataset-unique id number assigned to the question
			* answerChoices: possible answer options- the number and form will depend on the question type
			* correctAnswer: the correct answer given for the question
			* imagePath: relative path to image file from the dataset root directory.

	* topics:
		* topic name: name string, e.g. Carboniferous Period, Cenozoic Era, Cretaceous Period
		* content: instructional content extracted from source material
			* globalID: dataset-unique id number assigned to the topic
			* text: paragraph sized text blurb introducing and explaining topic
			* figures: diagram images associated with topic, if any
			* caption: image caption
			* imagePath: relative path to image file from the dataset root directory.
			* orderID: order of appearance within lesson

	* adjunctTopics:
		* section type: vocabulary, summary, introduction, apply concepts, etc.
		* content: instructional content extracted from source material
		* text: paragraph sized text blurb introducing and explaining topic
		* figures: diagram images associated with topic, if any
			* caption: image caption
			* imagePath: relative path to image file from the dataset root directory.
			* orderID: order of appearance within lesson

	* diagramAnnotations
		* diagram name: name of diagram image file (matching imageName field in
			diagramQuestions and instructionalDiagrams)
			* A list of groundtruth diagram text boxes with
				* text: human transcribed text
				* rectangle: location of text bounding box on the image in the form:
				[lower left coordinate, upper right coordinate]

	* instructionalDiagrams
		* diagram name:
			* "imagePath": relative path to image file from the dataset root directory.
        	* "imageName": name of the diagram image file,
			* "rawText": human written description of diagram,
			* "processedText": normalized text of the description

	* metaLessonID: an identifier that links lessons that are similar in their concepts and subject matter. This is to prevent 	  similar material being placed in both test and train.

## Citation
If you find this dataset helful in your work, please cite:

@inproceedings{Kembhavi2017tqa
title={Are You Smarter Than A Sixth Grader? Textbook Question Answering for Multimodal Machine Comprehension},
author={Aniruddha Kembhavi and Minjoon Seo and Dustin Schwenk and Jonghyun Choi and Ali Farhadi and Hannaneh Hajishirzi},
booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2017}
}
