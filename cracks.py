from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

# define an ontology to map class names to our OWLv2 prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
classes = ["crack"]

base_model = GroundedSAM(
    ontology=CaptionOntology(
        {
            "crack": "crack"
        }
    )
)

results = base_model.predict("crack.png")

image = cv2.imread("crack.png")

plot(
    image=image,
    detections=results,
    classes=[classes[i] for i in results.class_id],
)


print(results)
