
US Army Corps of Engineers eHydro Hydrographic Survey

About the Survey.zip File

Survey.zip File Contents

The Survey .zip file can contain the following files. (Many of these files are optional.) The
first part of each filename is the survey name (for example, "CL_28_VBR_20180530_CS", where
"CL_28_VBR" is the eHydro name of the channel, "20180530" is the date, and "CS" is an
optional description).

Extension:   .XYZ
Description: Thinned XYZ file
Req/Opt:     Required
Example:     CL_28_VBR_20180530_CS.XYZ

Extension:   _A.XYZ
Description: High-density XYZ file
Req/Opt:     Optional
Example:     CL_28_VBR_20180530_CS_A.XYZ

Extension:   _FULL.XYZ
Description: Very high-density XYZ file; not used in the desktop code for processing
Req/Opt:     Optional
Example:     CL_28_VBR_20180530_CS_FULL.XYZ

Extension:   .XML
Description: Metadata file
Req/Opt:     Optional
Example:     CL_28_VBR_20180530_CS.XML

Extension:   .XYZH
Description: XYZ header file
Req/Opt:     Optional
Example:     CL_28_VBR_20180530_CS.XYZH

Extension:   _SCCR.txt
Description: Survey Channel Condition Report
Req/Opt:     Required for coastal Districts with defined channels
Example:     CL_28_VBR_20180530_CS_SCCR.txt

Extension:   .PDF
Description: Chart PDF files; either a single multiple-page PDF file or multiple PDF files
Req/Opt:     Required
Example:     CL_28_VBR_20180530_CS.PDF

Extension:   .DAT
Description: XYZ file with a Z value the same as the labels on the chart
Req/Opt:     Required
Example:     CL_28_VBR_20180530_CS.DAT

Extension:   _ElevationContour.kmz
Description: ElevationContour feature in KMZ format
Req/Opt:     Optional
Example:     CL_28_VBR_20180530_CS_ElevationContour.kmz

Extension:   _ShoalPolygon.kmz
Description: ShoalPolygon feature in KMZ format
Req/Opt:     Optional
Example:     CL_28_VBR_20180530_CS_ShoalPolygon.kmz

Extension:   _SurveyPoint.kmz
Description: SurveyPoint feature in KMZ format
Req/Opt:     Optional
Example:     CL_28_VBR_20180530_CS_SurveyPoint.kmz

Extension:   _TIN.kmz
Description: TIN feature in KMZ format
Req/Opt:     Optional
Example:     CL_28_VBR_20180530_CS_TIN.kmz

Extension:   _SHP.zip
Description: Zipped directory with contour_lines, shoaling_polygons, and sounding_points in shapefile format
Req/Opt:     Optional
Example:     CL_28_VBR_20180530_CS_SHP.zip

Extension:   _tin
Description: TIN feature in a directory
Req/Opt:     Required
Example:     CL_28_VBR_20180530_CS_tin

Extension:   .gdb
Description: Survey geodatabase (see the lists below for the tables and features included)
Req/Opt:     Required
Example:     CL_28_VBR_20180530_CS.gdb



Tables in the Survey.gdb File

Table:       SPQ_Numbers
Description: Summary planning quantities (volume estimates) by quarter
Req/Opt:     Required for coastal Districts with defined channels

Table:       Channel_Availability_Table
Description: Shoalest depth by quarter
Req/Opt:     Required for coastal Districts with defined channels

Table:       version
Description: eHydro version number
Req/Opt:     Required


Features in the Survey.gdb File

Feature:     Bathymetry_Vector
Description: Polygon of the depth of the surveyed area
Req/Opt:     Required

Feature:     SurveyJob
Description: Polygon of the surveyed area
Req/Opt:     Required

Feature:     ElevationContour
Description: Contour lines on the chart
Req/Opt:     Required

Feature:     ElevationContour_ALL
Description: All of the Contour lines for the survey
Req/Opt:     Required

Feature:     Channel_Availability
Description: Shoalest survey points by quarter
Req/Opt:     Required for coastal Districts with defined channels

Feature:     ShoalPolygon
Description: Polygon of channel area above working depth
Req/Opt:     Required for coastal Districts with defined channels

Feature:     SurveyPoint
Description: XYZ file in GIS
Req/Opt:     Required

Feature:     SurveyPoint_HD
Description: _A.XYZ file in GIS
Req/Opt:     Optional

Feature:     ppxyz_points
Description: Extra XYZ points in GIS
Req/Opt:     Optional

