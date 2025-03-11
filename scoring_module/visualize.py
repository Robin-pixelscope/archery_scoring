import cv2


# grouping, total_score 추가 고려
class TargetVisualizer:
    def __init__(self, center_x, center_y):
        """
        {Number} center_x - The x-coordinate of the target center
        {Number} center_y - The y-coordinate of the target center
        """
        self.center_x = center_x
        self.center_y = center_y

    def draw_center_cross(self, img):
        """
        Draw a blue cross at the target's center coordinates.

        Parameters:
            {Numpy.array} img - The image on which to draw
        """
        color = (255, 0, 0)  # Blue
        cross_size = 20  # Length of each arm of the cross
        thickness = 2

        # Draw horizontal line
        cv2.line(
            img,
            (self.center_x - cross_size, self.center_y),
            (self.center_x + cross_size, self.center_y),
            color,
            thickness,
        )

        # Draw vertical line
        cv2.line(
            img,
            (self.center_x, self.center_y - cross_size),
            (self.center_x, self.center_y + cross_size),
            color,
            thickness,
        )

    def draw_arrow_hits(self, img, hits):
        """
        Draw red circles for detected arrow hits and display scores.

        Parameters:
            {Numpy.array} img - The image on which to draw
            {List} hits - A list of dictionaries with keys "point" (coordinates) and "score" (score)
                          Example: [{'point': (x, y), 'score': 10}, ...]
        """
        circle_color = (0, 0, 255)  # Red
        outline_color = (0, 0, 0)  # Black outline
        circle_radius = 10
        thickness = 5  # Filled circle

        for hit in hits:
            x, y = hit["point"]
            score_2f = f"{hit['score']:.2f}"
            score = str(score_2f)

            # Draw the red circle
            cv2.circle(img, (x, y), circle_radius, circle_color, thickness)

            # Draw black outline
            cv2.circle(img, (x, y), circle_radius + 2, outline_color, 2)

            # Draw the score above the hit point
            cv2.putText(
                img,
                score,
                (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                5,
            )
            cv2.putText(
                img,
                score,
                (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

    def visualize(self, img, hits):
        """
        Visualize the entire setup with center cross and hits.

        Parameters:
            {Numpy.array} img - The image on which to draw
            {List} hits - A list of hit dictionaries as described above
        """
        self.draw_center_cross(img)
        self.draw_arrow_hits(img, hits)
        return img


# # Example usage:
# # center coordinates of the target
# center_x, center_y = 320, 240

# # Create the visualizer
# visualizer = TargetVisualizer(center_x, center_y)

# # Mock image (substitute with actual image)
# img = cv2.imread(
#     "/home/robin/code/ultralytics/archery/dataset/Data_V1/images/20240912153113_A03_0.jpg"
# )  # Load an image

# # Example hit points with scores
# hits = [
#     {"point": (340, 260), "score": 9},
#     {"point": (360, 220), "score": 7},
#     {"point": (300, 230), "score": 10},
# ]

# # Draw the visualization
# output_img = visualizer.visualize(img, hits)

# # Display the image
# cv2.imshow("Visualization", output_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
