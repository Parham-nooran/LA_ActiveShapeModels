from cgi import print_directory
from turtle import back
import numpy as np
import cv2
import dlib
import time
from scipy.spatial import ConvexHull


# Phase 1 --------------------------------------------------------------------------------

numberOfImages = 39
images = []
faces = []
Xs = []


def get_index_of(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def average_face(X):
    m = np.sum(X, axis=0)//68
    for x in X:
        x -= m

def getCopy(images):
    copy = []
    for image in images:
        copy.append(image.copy())
    return copy 


def register_affine(X, X1):
    X1Inner = X1.T @ X1 # TODO replace with inner or dot
    sx1 = X1Inner[0, 0]
    sx1y1 = X1Inner[1, 0]
    sy1 = X1Inner[1, 1]
    X1XInner = X1.T @ X # TODO replace with inner or dot
    sx1x = X1XInner[0, 0]
    sy1x = X1XInner[1, 0]
    sy1y = X1XInner[1, 1]
    sx1y = X1XInner[0, 1]
    
    A = np.linalg.solve(np.array([[sx1, sx1y1, 0, 0], [sx1y1, sy1, 0, 0],
     [0, 0, sx1, sx1y1], [0, 0, sx1y1, sy1]]),
    np.array([sx1x, sy1x, sx1y, sy1y]))

    return (np.reshape(A, (2, 2)) @ X1.T).T


def register_similarity(X, X1):
    X1Inner = X1.T @ X1 # TODO replace with inner or dot
    sx1 = X1Inner[0, 0]
    sy1 = X1Inner[1, 1]

    X1XInner = X1.T @ X # TODO replace with inner or dot
    sx1x = X1XInner[0, 0]
    sy1x = X1XInner[1, 0]
    sy1y = X1XInner[1, 1]
    sx1y = X1XInner[0, 1]

    a = (sx1x + sy1y) / (sx1 + sy1)
    b = (sx1y - sy1x) / (sy1 + sx1) 

    A = np.array([[a,-b], [b, a]])
    return (X1 @ np.reshape(A, (2, 2)).T)



for i in range(1, numberOfImages + 1):
    images.append(cv2.imread(f'images1/{i}.jpg', cv2.IMREAD_UNCHANGED))

# for i in range(1, numberOfImages + 1):
#     images.append(cv2.imread(f'images/p{i}.jpg', cv2.IMREAD_UNCHANGED))

images_copy = getCopy(images)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


for image in images:
    faces.append(detector(image)[0])


for i in range(numberOfImages):
    landmarks = predictor(images[i], faces[i])
    Xs.append(np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(landmarks.num_parts)]))

# Phase 1 - Part 1 : Showing The Average Face on Both The Neutral Face and a White Page 

for i in range(numberOfImages):
    key = 0
    for x in Xs[i]:
        cv2.circle(images[i], (x[0], x[1]), 2, (0, 255, 0), -1)
    cv2.imshow(f'Landmarks of Face #{i}', images[i])
    key = cv2.waitKey(0)
    if key == 27 or key == ord('q'):
        break

images = getCopy(images_copy)
cv2.destroyAllWindows()

# Phase 1 - Part 1 : Showing All Faces With Landmarks

for i in range(numberOfImages):
    key = 0
    for x in Xs[i]:
        cv2.circle(images[0], (x[0], x[1]), 2, (0, 255, 0), -1)
    cv2.imshow(f'Landmarks of All Faces on The Neutral Face', images[0])
    key = cv2.waitKey(0)
    if key == 27 or key == ord('q'):
        break
images = getCopy(images_copy)
cv2.destroyAllWindows()

Xs_copy = np.array(Xs.copy(), dtype=np.int32)

# (Phase 2 - 1)

for i in range(numberOfImages):
    average_face(Xs[i])

for i in range(1, numberOfImages):
    Xs[i] = register_similarity(Xs[0], Xs[i]).astype(np.int32)

# for i in range(1, numberOfImages):
#     Xs[i] = register_affine(Xs[0], Xs[i]).astype(np.int32)

h, w, c = images[0].shape

# Phase 1 - Part 2 : Showing All Faces Registered To The First

for i in range(numberOfImages):
    key = 0
    for x in Xs[i]:
        cv2.circle(images[0], (x[0]+w//2, x[1]+h//2), 2, (0, 255, 0), -1)
    cv2.imshow(f'Landmarks of All Faces After Registration, on The Neutral Face', images[0])
    key = cv2.waitKey(0)
    if key == 27 or key == ord('q'):
        break

# images = getCopy(images_copy)
# cv2.destroyAllWindows()


img = np.zeros([h,w,3])
img[:] = 255
temp = img.copy()


Xs = np.array(Xs)


Xs = Xs.reshape((numberOfImages, 136, 1))

# (Phase 2 - 2) 
mu = np.sum(Xs, axis=0)//numberOfImages
# mu = Xs.mean(axis=1)


# Phase 1 - Part 3 : Showing The Average Face on Both The Neutral Face and a White Page 


for x in mu.reshape((68, 2)):
    cv2.circle(images[0], (x[0]+w//2, x[1]+h//2), 2, (0, 255, 0), -1)
cv2.imshow('Landmarks of The Average Face on The Neutral Face', images[0])
key = cv2.waitKey(0)

images = getCopy(images_copy)
cv2.destroyAllWindows()


for x in mu.reshape((68, 2)):
    cv2.circle(img, (x[0]+w//2, x[1]+h//2), 2, (0, 255, 0), -1)
cv2.imshow('Landmarks of The Average Faces on a White Page', img)
key = cv2.waitKey(0)
img = temp.copy()
cv2.destroyAllWindows()


# Phase 2 --------------------------------------------------------------------------------

K = 30
# Phase 2 - 3 : Subtracting The Average Face From The Other Faces
Z = Xs - mu.reshape((-1, 1))
# Phase 2 - 4 : Applying PCA to Compute The Principal Components, Using SVD
U, sigma, vh = np.linalg.svd(Z, full_matrices=False)
U = U.reshape((numberOfImages, 136)).T 
sigma = sigma[:K]
U = U[:, :K]
a = np.zeros(K).reshape(K, 1)
X = mu + U @ a
X = X.reshape((68, 2)).astype(np.int32)

# Phase 2 - Animating The First K Modes

img = temp.copy()
key = 0
for i in range(K):
    avals = np.linspace(-sigma[i], sigma[i], 5)
    for aval in avals:
        a[i] = aval

        X = mu + U @ a
        a[i] = 0

        X = X.reshape((68, 2)).astype(np.int32)

        for j in range(numberOfImages):
            for x in X:
                cv2.circle(img, (x[0]+w//2, x[1]+h//2), 2, (255, 0, 0), -1)
        
        cv2.imshow("Phase 2 - Different Sigmas On The Statistical Model of Face", img)
        key = cv2.waitKey(1000)
        
        img = temp.copy()

        if key == 27 or key == ord('q'):
            break
    if key == 27 or key == ord('q'):
        break
        
img = temp.copy()
cv2.destroyAllWindows()

# Phase 3 --------------------------------------------------------------------------------


vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False


count = 0
while rval:
    img = temp.copy()
    rval, frame = vc.read()
    #print(rval)
    count += 1
    if count % 3 != 1:
        pass
    else:
        faces = detector(frame, 1)
    
    cv2.imshow('', frame)

    if(len(faces) > 0):
        landmarks = predictor(frame, faces[0])
        X = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(landmarks.num_parts)])
        average_face(X)
        X = register_affine(mu.reshape((68, 2)), X)
        X = X.reshape((136, 1))
        pinv = np.linalg.pinv(U) # np.linalg.inv(np.dot(U.T,U)),U.T), (UTU)−1UT  is called the pseudo-inverse
        a_star = pinv.dot(X-mu)
        F = mu + U @ a_star
        F = F.reshape((68, 2)).astype(np.int32)
        for x in F:
            cv2.circle(img, (x[0]+w//2, x[1]+h//2), 2, (255, 0, 0), -1)

        cv2.imshow('Phase 3 - Deep Fake But Easier', img)
    
    key = cv2.waitKey(1)

    if key == 27 or key == ord('q'):
        break
    elif key == ord(' '):
        print('X = ', X)

img = temp.copy()
cv2.destroyAllWindows()


# Phase 4 --------------------------------------------------------------------------------


Xs = Xs_copy.copy()
images = getCopy(images_copy)

Xs = Xs.reshape((numberOfImages, 68, 2))

# Neutral Face

neutral_face_points = np.array(Xs[0], np.int32)
neutral_face_convexhull = cv2.convexHull(neutral_face_points) 
img_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img_gray) 

cv2.fillConvexPoly(mask , neutral_face_convexhull, 255)

face_image_1 = cv2.bitwise_and(images[0], images[0], mask = mask)

# Delaunay tiangulation
rect = cv2.boundingRect(neutral_face_convexhull)
subdiv = cv2.Subdiv2D(rect)

for p in neutral_face_points:
    subdiv.insert((int(p[0]), int(p[1])))

triangles = np.array(subdiv.getTriangleList(), dtype=np.int32)

triangles_indices = []

for t in triangles:
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])
    index_pt1 = np.where((neutral_face_points == pt1).all(axis=1))
    index_pt1 = get_index_of(index_pt1)
    index_pt2 = np.where((neutral_face_points == pt2).all(axis=1))
    index_pt2 = get_index_of(index_pt2)
    index_pt3 = np.where((neutral_face_points == pt3).all(axis=1))
    index_pt3 = get_index_of(index_pt3)


    if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
        triangle = [index_pt1, index_pt2, index_pt3]
        triangles_indices.append(triangle)



vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False


count = 0
while rval:
    img = temp.copy()
    rval, frame = vc.read()
    #print(rval)
    count += 1
    if count % 3 != 1:
        pass
    else:
        faces = detector(frame, 1)
    
    cv2.imshow('', frame)

    new_statistical = np.zeros_like(images[0])
    if(len(faces) > 0):
        landmarks = predictor(frame, faces[0])
        X = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(landmarks.num_parts)])
        average_face(X)
        X = register_affine(mu.reshape((68, 2)), X)
        X = X.reshape((136, 1))
        pinv = np.linalg.pinv(U) # np.linalg.inv(np.dot(U.T,U)),U.T), (UTU)−1UT  is called the pseudo-inverse
        a_star = pinv.dot(X-mu)
        F = mu + U @ a_star
        F = F.reshape((68, 2)).astype(np.int32)
        
        for x in F:
            x[0] += 25

        statistical_face_points = F.reshape((68, 2)).astype(int).copy()
        img = temp.copy()
        

        for x in statistical_face_points:
            x[0] += w//2
            x[1] += h//2


        for triangle_index in triangles_indices:

            tr1_pt1 = neutral_face_points[triangle_index[0]]
            tr1_pt2 = neutral_face_points[triangle_index[1]]
            tr1_pt3 = neutral_face_points[triangle_index[2]]

            netural_face_triangle = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
            neutral_face_rect = cv2.boundingRect(netural_face_triangle)

            x, y, width , height = neutral_face_rect
            neutral_cropped = images[0][y:y+height, x:x+width]
            neutral_cropped_mask = np.zeros((height, width), np.uint8)

            neutral_cropped_points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                                [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                                [tr1_pt3[0] - x, tr1_pt3[1] - y]])

            cv2.fillConvexPoly(neutral_cropped_mask, neutral_cropped_points, 255)
            neutral_cropped = cv2.bitwise_and(neutral_cropped, neutral_cropped, mask=neutral_cropped_mask)


            tr2_pt1 = statistical_face_points[triangle_index[0]]
            tr2_pt2 = statistical_face_points[triangle_index[1]]
            tr2_pt3 = statistical_face_points[triangle_index[2]]
            

            statistical_face_triangle = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
            statistical_face_rect = cv2.boundingRect(statistical_face_triangle)

            x, y, width2 , height2 = statistical_face_rect
            statistical_cropped = img[y:y+height2, x:x+width2]

            statistical_cropped_mask = np.zeros((height2, width2), np.uint8)

            statistical_cropped_points = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                                [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                                [tr2_pt3[0] - x, tr2_pt3[1] - y]])

            cv2.fillConvexPoly(statistical_cropped_mask, statistical_cropped_points, 255)
            statistical_cropped = cv2.bitwise_and(statistical_cropped, statistical_cropped,
                                                    mask=statistical_cropped_mask)
            
            neutral_cropped_points = np.float32(neutral_cropped_points)
            statistical_cropped_points = np.float32(statistical_cropped_points)

            matrix = cv2.getAffineTransform(neutral_cropped_points, statistical_cropped_points)

            warped_triangle = cv2.warpAffine(neutral_cropped, matrix, (width2, height2))

            triangle_area = new_statistical[y:y+height2, x:x+width2]
            triangle_area = cv2.add(triangle_area, warped_triangle)

            new_statistical[y:y+height2, x:x+width2] = triangle_area


        new_statistical_gray = cv2.cvtColor(new_statistical, cv2.COLOR_BGR2GRAY)
        correct, background = cv2.threshold(new_statistical_gray, 1, 255, cv2.THRESH_BINARY_INV)
        head_mask = np.bitwise_not(background)
        background = cv2.bitwise_and(images[0], images[0], mask=background)
        result = cv2.add(background, new_statistical)
        # (x, y, w, h) = cv2.boundingRect(neutral_face_convexhull)
        # center_face = (int((x + x + w)/2), int((y + y + h) / 2))

        # seamlessclone = cv2.seamlessClone(result, images[1], head_mask, center_face, cv2.MIXED_CLONE)
        
        cv2.imshow('Phase 4, Result', result)
        # cv2.imshow('Phase 4, Head mask', head_mask)
        # cv2.imshow('Phase 4, Background', background)
        # cv2.imshow('Phase 4, New face', new_statistical)
        # cv2.imshow('Phase 4, Seamless Clone', seamlessclone)


    key = cv2.waitKey(1)

    if key == 27 or key == ord('q'):
        break
    elif key == ord(' '):
        print('X = ', X)




cv2.destroyAllWindows()



# Phase 5 --------------------------------------------------------------------------------

# x1 = face.left()
# y1 = face.top()
# x2 = face.right()
# y2 = face.bottom()

# cv2.rectangle(images[0], (x1, y1), (x2, y2), (0, 255, 0), 3)




# for x in Xs[0]:
#     cv2.circle(images[0], (x[0], x[1]), 2, (255, 0, 0), -1)

# cv2.imshow("Phase 4 - neutral Face's Convex", face_image_1)
# cv2.imshow('Phase 4 - Mask', mask)


# Statistical Face 

# statistical_face_points = (mu + (U @ a)).reshape((68, 2)).astype(int).copy()
# img = temp.copy()
# new_statistical = np.zeros_like(images[0])


# for x in statistical_face_points:
#     x[0] += w//2
#     x[1] += h//2


# for triangle_index in triangles_indices:

#     tr1_pt1 = neutral_face_points[triangle_index[0]]
#     tr1_pt2 = neutral_face_points[triangle_index[1]]
#     tr1_pt3 = neutral_face_points[triangle_index[2]]

#     netural_face_triangle = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
#     neutral_face_rect = cv2.boundingRect(netural_face_triangle)

#     x, y, width , height = neutral_face_rect
#     neutral_cropped = images[0][y:y+height, x:x+width]
#     neutral_cropped_mask = np.zeros((height, width), np.uint8)

#     neutral_cropped_points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
#                                         [tr1_pt2[0] - x, tr1_pt2[1] - y],
#                                         [tr1_pt3[0] - x, tr1_pt3[1] - y]])

#     cv2.fillConvexPoly(neutral_cropped_mask, neutral_cropped_points, 255)
#     neutral_cropped = cv2.bitwise_and(neutral_cropped, neutral_cropped, mask=neutral_cropped_mask)

#     # cv2.line(images[0], tr1_pt1, tr1_pt2, (0, 255, 0), 1)
#     # cv2.line(images[0], tr1_pt2, tr1_pt3, (0, 255, 0), 1)
#     # cv2.line(images[0], tr1_pt3, tr1_pt1, (0, 255, 0), 1)


#     tr2_pt1 = statistical_face_points[triangle_index[0]]
#     tr2_pt2 = statistical_face_points[triangle_index[1]]
#     tr2_pt3 = statistical_face_points[triangle_index[2]]

#     # cv2.line(img, tr2_pt1, tr2_pt2, (0, 0, 255), 1)
#     # cv2.line(img, tr2_pt2, tr2_pt3, (0, 0, 255), 1)
#     # cv2.line(img, tr2_pt3, tr2_pt1, (0, 0, 255), 1)


#     statistical_face_triangle = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
#     statistical_face_rect = cv2.boundingRect(statistical_face_triangle)

#     x, y, width2 , height2 = statistical_face_rect
#     statistical_cropped = img[y:y+height2, x:x+width2]

#     statistical_cropped_mask = np.zeros((height2, width2), np.uint8)

#     statistical_cropped_points = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
#                                         [tr2_pt2[0] - x, tr2_pt2[1] - y],
#                                         [tr2_pt3[0] - x, tr2_pt3[1] - y]])

#     cv2.fillConvexPoly(statistical_cropped_mask, statistical_cropped_points, 255)
#     statistical_cropped = cv2.bitwise_and(statistical_cropped, statistical_cropped,
#                                              mask=statistical_cropped_mask)
    
#     neutral_cropped_points = np.float32(neutral_cropped_points)
#     statistical_cropped_points = np.float32(statistical_cropped_points)

#     matrix = cv2.getAffineTransform(neutral_cropped_points, statistical_cropped_points)

#     warped_triangle = cv2.warpAffine(neutral_cropped, matrix, (width2, height2))

#     triangle_area = new_statistical[y:y+height2, x:x+width2]
#     triangle_area = cv2.add(triangle_area, warped_triangle)

#     new_statistical[y:y+height2, x:x+width2] = triangle_area


# new_statistical_gray = cv2.cvtColor(new_statistical, cv2.COLOR_BGR2GRAY)
# correct, background = cv2.threshold(new_statistical_gray, 1, 255, cv2.THRESH_BINARY_INV)
# background = cv2.bitwise_and(img, img, mask=background)
# # result = cv2.add(background, new_statistical, dtype=np.int32)


# cv2.imshow('Phase 4 - neutral Face', images[0])
# cv2.imshow('Phase 4, result', new_statistical)