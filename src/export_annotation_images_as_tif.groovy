// Usage:
// Open Qupath
// Automate > Show script editor
// Load the script as user script in Qupath for a given project
// Line 12: Enter corresponding Qupath Project Base Directory
// Open Project in Qupath, select first Image and run the script by pressing: Run -> Run for Project
//   Pressing in "Script Editor" window (opened by Automate > Show script editor)
//   After pressing select all images and confirm using ok button
// Results are in ROIsTif directory in QuPath project root

import qupath.lib.images.ImageData

def server = getCurrentServer()

def baseImageName = getProjectEntry().getImageName()

path = buildFilePath("/home/kikou/Uni/organoid/annotation", 'ROIsTif')
mkdirs(path)

getAnnotationObjects().eachWithIndex{it, x->
        println("Working on: "+it)
	def roi = it.getROI()
	def requestROI = RegionRequest.createInstance(server.getPath(), 1, roi)
	currentImagePath = buildFilePath(PROJECT_BASE_DIR, 'ROIsTif',baseImageName +it+'_'+ x + '.tif')
	writeImageRegion(server, requestROI, currentImagePath)
}
print 'Done!'