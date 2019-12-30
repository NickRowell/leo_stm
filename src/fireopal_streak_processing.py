"""
Created on Fri Dec 20 2019

@author: Nick Rowell, after Maureen Cohen

Purpose: this script performs streak processing and satellite trail assembly on a
bunch of streaks detected at the image processing stage of the ROE FireOPAL pipeline.

"""

from fireopal_settings import *
import numpy as np, datetime
import smtplib
from os.path import basename
from email.message import EmailMessage
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate

class Streak:
    
    """ 
    A Streak object represents a distinct streak found in an Image object.
    It is characterised by the image and celestial coordinates of each end of
    the streak, and two times that are the opening and closing times of the shutter.
    At first, the correspondence between the streak ends and the times is not known.
    Once the streak direction has been resolved, the coordinates with subscript 1
    correspond to the shutter opening, i.e. streak start.
    
    """
    def __init__(self, line):
        columns = line.split(',') 

        # TODO no longer use time_a and time_b recorded in the streak. These are worked out on the fly from the filename.

        # file, ra1, dec1, x1, y1, ra2, dec2, x2, y2, time_a, time_b
        self.filename = columns[0]
        self.ra1 = float(columns[1])
        self.dec1 = float(columns[2])
        self.x1 = float(columns[3])
        self.y1 = float(columns[4])
        self.ra2 = float(columns[5])
        self.dec2 = float(columns[6])
        self.x2 = float(columns[7])
        self.y2 = float(columns[8])
        # Get streak end point times by extracting shutter open/close times from filename
        self.time_open , self.time_close = get_shutter_open_close_datetimes(self.filename)

    def __repr__(self):
        return "Streak()"
    
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def swap_ends(self):
        self.x1, self.x2 = self.x2, self.x1
        self.y1, self.y2 = self.y2, self.y1
        self.ra1, self.ra2 = self.ra2, self.ra1
        self.dec1, self.dec2 = self.dec2, self.dec1

    def to_string(self):

        return '{:.3f},{:.3f},{},{:.3f},{:.3f},{},'.format(self.ra1, self.dec1, self.time_open, self.ra2, self.dec2, self.time_close)

def get_shutter_open_close_datetimes(filename):

    # TODO are there any fudge factors that need to be applied here?

    # 005_2019-12-17_171458_A_DSC_0187.NEF -> 2019-12-17_171458
    datestamp = "".join(list(str(filename))[4:21])
    time_open = datetime.datetime.strptime(datestamp, '%Y-%m-%d_%H%M%S')
    time_close = time_open + datetime.timedelta(seconds=5)
    return time_open, time_close

def normal_line(x1, y1, x2, y2):

    """
    Calculates the normal representation of the line from two points.
    Can handle vertical lines.

    Inputs: two sets of (x,y) pixel coordinates
    Outputs: orientation (theta [deg]) and distance to origin (d [pixels])

    The orientation angle is measured anticlockwise from the x axis
    and lies in the range [-180:180]. The distance to origin is always
    positive.
    
    """

    dx = x2-x1
    dy = y2-y1
    n = np.sqrt(dx*dx + dy*dy)

    # Compute the unit normal to the line. At this stage it may point
    # towards or away from the origin.
    nx = dy/n
    ny = -dx/n

    # Compute the distance to the origin
    d = x1 * nx + y1 * ny

    # If this is negative then flip the direction of the normal
    if d < 0:
        nx *= -1
        ny *= -1
        d *= -1

    # Absolute value of angle measured from x axis to the normal
    theta = np.arccos(nx)

    # Correct sign
    if ny < 0:
        theta *= -1

    # Convert to degrees
    theta = np.degrees(theta)

    return theta, d

def streak_processing(output):

    ############################
    #
    #     Load and map data
    #
    ############################

    # Opens txt file where data from Fire Opal image processing is stored
    data = open(output + 'streaks_data.txt', 'r')

    # Loads each line of the file into a list of strings
    lines = data.readlines()

    # Create empty dictionary to map images to lists of streaks. This contains all of the streak activity for the night.
    activity = {}

    for line in lines:

        # Strip the \n off the end of the line
        line = line.rstrip('\n')

        # Parse the Streak from the string of text
        streak = Streak(line)

        # Check if we already found a streak in this image
        if streak.filename not in activity:
            # Create new entry in dictionary; add this streak as the first
            activity[streak.filename] = [streak]
        else:
            # Multiple streaks in the same image
            activity[streak.filename].append(streak)


    # Build nested list to store satellite trails
    satellites = []

    ############################
    #
    #       Trail assembly
    #
    ############################

    # Loop over the images. We only look for matching streaks forwards in time.
    for idx, image in enumerate(sorted(activity)):

        # Loop over the streaks in this image
        for streak in activity[image]:

            # The set of streaks that have been matched to this one comprise one satellite
            satellite = [streak]

            # Get the line parameters for this streak
            theta, r = normal_line(streak.x1, streak.y1, streak.x2, streak.y2)

            # TODO: if the streak cuts the image edge then the path hypotheses will be wrong!
            # These streaks need special handling.

            # Determine two hypotheses for the position vector as a function of time for this streak
            x1 = streak.x1
            y1 = streak.y1
            x2 = streak.x2
            y2 = streak.y2

            # datetime.datetime objects
            ta = streak.time_open
            tb = streak.time_close

            # Lambda expressions return (x,y) as a function of t for each path hypothesis. For streaks that cut the image edge this will give
            # a lower bound on the velocity across the image, so is still useful for assessing visibility, i.e. there's no risk that the path
            # will predict the satellite has moved out of the image when it's still inside it.
            path1 = lambda t : (x1 + (t - ta).total_seconds() * (x2 - x1) / (tb - ta).total_seconds() , y1 + (t - ta).total_seconds() * (y2 - y1) / (tb - ta).total_seconds())
            path2 = lambda t : (x1 + (t - tb).total_seconds() * (x2 - x1) / (ta - tb).total_seconds() , y1 + (t - tb).total_seconds() * (y2 - y1) / (ta - tb).total_seconds())

            # Enter the path hypotheses into a list
            paths = [path1, path2]

            # Check if either find a matching streak in the subsequent images
        
            # Loop over subsequent images until there's no more chance of finding a matching streak
            for image2 in sorted(activity)[idx+1:]:

                # Get shutter open and close time of this image
                time_open, time_close = get_shutter_open_close_datetimes(image2)

                # If all remaining path hypotheses put the satellite far outside the image then we're done searching
                satellite_present = False
                for path in paths:
                    # Check if this path predicts satellite is present in image at the shutter opening time. No need
                    # to check for shutter closing.
                    x_o, y_o = path(time_open)
                    if(x_o > -margin and x_o < image_width+margin and y_o > -margin and y_o < image_height+margin):
                        # If satellite is moving in this direction then it should be visible at shutter opening
                        satellite_present = True

                if satellite_present == False:
                    # Break to next streak
                    break

                # Loop over streaks in this image
                for streak2 in activity[image2]:

                    # Check streak (theta, r) line parameters to determine if it's a possible match
                    theta2, r2 = normal_line(streak2.x1, streak2.y1, streak2.x2, streak2.y2)

                    if abs(theta - theta2) < angle_match_threshold and abs(r - r2) < distance_match_threshold:

                        # Found a candidate matching streak.

                        # TODO Determine which path hypothesis is correct

                        # TODO Eliminate the incorrect hypothesis from the array, if possible

                        # Enter this streak into the list of matching streaks
                        satellite.append(streak2)

                        # Remember to eliminate the streak from the map so that we don't process it later on
                        activity[image2].remove(streak2)

            satellites.append(satellite)

    ############################
    #
    #       Write outputs
    #
    ############################

    print('Found ' + str(len(satellites)) + ' satellites')

    # XXX Debugging
    #for satellite in satellites:
    #    print('Streaks = ' + str(len(satellite)))
    #    for streak in satellite:
    #        print(streak.filename + ' ' + str(streak.x1) + ' ' + str(streak.y1))

    # Post-process satellites to resolve direction and assign times to the streak end points.
    # TODO can the resolving of end point times be done during trail assembly?

    # Create output file
    # TODO: add date
    resultsFile = output + 'satellites.txt'
    with open(resultsFile, 'a+') as txtFile:
        for idx, streaks in enumerate(satellites):

            # If there's two or more streaks then we can resolve direction
            if len(streaks) > 1:
                # Use the first two streaks to determine the direction
                streak1 = streaks[0]
                streak2 = streaks[1]

                x1 = streak1.x1
                y1 = streak1.y1
                x2 = streak1.x2
                y2 = streak1.y2

                # datetime.datetime objects
                to = streak1.time_open
                tc = streak1.time_close

                # Lambda expressions return (x,y) as a function of t for each path hypothesis

                # Path1 assumes (x1,y1) occurs at shutter opening and (x2,y2) at shutter closing.
                path1 = lambda t : (x1 + (t - to).total_seconds() * (x2 - x1) / (tc - to).total_seconds() , y1 + (t - to).total_seconds() * (y2 - y1) / (tc - to).total_seconds())

                # Path2 assumes (x2,y2) occurs at shutter opening and (x1,y1) at shutter closing.
                path2 = lambda t : (x1 + (t - tc).total_seconds() * (x2 - x1) / (to - tc).total_seconds() , y1 + (t - tc).total_seconds() * (y2 - y1) / (to - tc).total_seconds())

                # Observed midpoint of streak2
                xm = (streak2.x1 + streak2.x2)/2
                ym = (streak2.y1 + streak2.y2)/2

                # Use both hypotheses to predict the location of the midpoint of the second streak
                dt = (streak2.time_close - streak2.time_open) / 2
                t_mid = streak2.time_open + dt

                xm_p1, ym_p1 = path1(t_mid)
                xm_p2, ym_p2 = path2(t_mid)

                # Differences between predicted & observed midpoints
                offset_xp1 = xm_p1 - xm
                offset_yp1 = ym_p1 - ym
                offset_p1 = np.sqrt(offset_xp1*offset_xp1 + offset_yp1*offset_yp1)

                offset_xp2 = xm_p2 - xm
                offset_yp2 = ym_p2 - ym
                offset_p2 = np.sqrt(offset_xp2*offset_xp2 + offset_yp2*offset_yp2)

                # XXX Debugging
                #print('streak1: x1 y1 = ' + str(x1) + ' ' + str(y1))
                #print('streak1: x2 y2 = ' + str(x2) + ' ' + str(y2))
                #print('streak2: x1 y1 = ' + str(streak2.x1) + ' ' + str(streak2.y1))
                #print('streak2: x2 y2 = ' + str(streak2.x2) + ' ' + str(streak2.y2))
                #print('streak2: xm ym = ' + str(xm) + ' ' + str(ym))
                #print('offset_p1 = ' + str(offset_p1))
                #print('offset_p2 = ' + str(offset_p2))

                # Smallest error indicates the correct path. Resolve the ends of streak1 so that (x1,y1) occurs
                # at shutter opening and (x2,y2) at shutter closing.
                if offset_p1 < offset_p2:
                    # Path1 is correct. (x1,y1) occurs at shutter opening and (x2,y2) at shutter closing.
                    # No action to take
                    pass
                else:
                    # Path 2 is correct. (x2,y2) occurs at shutter opening and (x1,y1) at shutter closing.
                    # Need to swap the order of the points.
                    streak1.swap_ends()

                # Now loop over streak 2+ and resolve order of ends
                for streak in streaks[1:]:
                    # The point on this streak with the shortest distance to the end of the first streak is the streak start. Other point is the streak end
                    d1 = np.sqrt((streak.x1 - streak1.x2)*(streak.x1 - streak1.x2) + (streak.y1 - streak1.y2)*(streak.y1 - streak1.y2))
                    d2 = np.sqrt((streak.x2 - streak1.x2)*(streak.x2 - streak1.x2) + (streak.y2 - streak1.y2)*(streak.y2 - streak1.y2))
                    if d1 < d2:
                        # Point 1 occurs at shutter opening; no need to swap
                        pass
                    else:
                        # Point 2 occurs at shutter opening; need to swap
                        streak.swap_ends()

                # Now loop over all streaks and write them out
                for streak in streaks:
                    txtFile.write(streak.to_string())
                txtFile.write('\n')

            # If there's just one streak then all we can provide the coordinates of the end points and the shutter open/close times
            else:
                txtFile.write(streaks[0].to_string())
                txtFile.write('\n')

    txtFile.close()

    # TODO: create some visualisation of the detected satellites

    # TODO: email to addresses given in the settings

    msg = MIMEMultipart()
    msg['Subject'] = 'FireOPAL results '
    msg['From']    = 'nr@roe.ac.uk'
    msg['To']      = 'nr@roe.ac.uk'
    msg['Date']    = formatdate(localtime=True)

    # TODO Insert correct date
    msg.attach(MIMEText('This email contains the results of the ROE FireOPAL pipeline for the night starting on XXXXX'))

    # Attach the satellites.xt file
    with open(resultsFile, "rb") as fil:
        part = MIMEApplication(fil.read(),Name=basename(resultsFile))
    part['Content-Disposition'] = 'attachment; filename="%s"' % basename(resultsFile)
    msg.attach(part)

    s = smtplib.SMTP('mail.roe.ac.uk', 25)
    #s.send_message(msg)
    s.quit()

if __name__ == "__main__":
    streak_processing(output)


