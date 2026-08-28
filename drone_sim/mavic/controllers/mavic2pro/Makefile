


# Do not modify the following: this includes Webots global Makefile.include
null :=
space := $(null) $(null)
WEBOTS_HOME_PATH?=$(subst $(space),\ ,$(strip $(subst \,/,$(WEBOTS_HOME))))
NO_FAT_BINARY = 1
PROCESSOR = arm64
INCLUDE = -I/opt/homebrew/include
LIBRARIES = /opt/homebrew/lib/libcjson.a
include $(WEBOTS_HOME_PATH)/resources/Makefile.include
