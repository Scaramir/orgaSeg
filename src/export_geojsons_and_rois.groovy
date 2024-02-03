// Usage:
// Open project in Qupath
// Automate > Show script editor
// Import this script in "Script Editor"
// Run the script by pressing: "Run..." -> "Run for Project"
//   After pressing, select all images and confirm using "ok" button
// Results are in exports/ directory in QuPath project root
//   There will be a ROIsTif/ and geojsons/ directory inside containing results

import qupath.lib.images.ImageData

def server = getCurrentServer()

def baseImageName = getProjectEntry().getImageName()

def exportPath = buildFilePath(PROJECT_BASE_DIR, 'export')

// export ROIs

def roisPath = buildFilePath(exportPath, 'ROIsTif')
mkdirs(roisPath)

getAnnotationObjects().eachWithIndex{it, x->
        println("Working on: "+it)
	def roi = it.getROI()
	def requestROI = RegionRequest.createInstance(server.getPath(), 1, roi)
	currentImagePath = buildFilePath(roisPath, baseImageName +it+'_'+ x + '.tif')
	writeImageRegion(server, requestROI, currentImagePath)
}

// export geojsons

def geojsonPath = buildFilePath(exportPath, 'geojsons')
mkdirs(geojsonPath)

def annotations = getAnnotationObjects()
exportObjectsToGeoJson(annotations,
                       buildFilePath(geojsonPath, baseImageName + ".geojson"),
                       "FEATURE_COLLECTION")


print 'Done! (' + baseImageName + ")"