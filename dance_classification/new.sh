#! /bin/bash

# ECHO COMMAND
echo Hello World!

#VARIABLES
#Uppercase by convention
# NAME="BRAD"
# echo "My name is $NAME"

#USER INPUT
#read -p "Enter your name: " NAME
#echo "Hello $NAME"

# if [ "$NAME" == "BRAD" ]
# then
#  echo "Your name is brad"
# elif ["$NAME" == "Jack"]
# then
#   echo "Your name is jack"
# else
#  echo "Your name is not brad"
# fi

#NUM1=3
#NUM2=5

# if [ "$NUM1" -gt "$NUM2" ]
#then
#  echo "$NUM1 is greater than $NUM2"
#else
#  echo "$NUM1 is less than $NUM2"
#fi

# FILE CONDITIONS
# FILE="test.txt"
# if [ -f "$FILE" ]
# then
#  echo "$FILE is a file"
# else
#  echo "$FILE is not a file"
# fi

read -p "Are you 21 or over? Y/N " ANSWER
case "$ANSWER" in
  [yY] | [yY][eE][sS]

