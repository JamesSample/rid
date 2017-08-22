# Riverine Inputs and Direct Discharges (RID) Programme (Elveovervåkingsprogrammet)

This repository includes code for **estimating pollutant loads to Norwegian coastal waters**. The work is part of a joint  monitoring programme under the [OSPAR Commission](https://www.ospar.org/) for the **Protection of the Marine Environment of the Northeast Atlantic**. The main purpose is to estimate total loads of selected pollutants draining annually to Convention waters from inland water bodies.

The Norwegian component of the programme involves collaboration between [NIVA](http://www.niva.no/), [NIBIO](http://www.nibio.no/) and [NVE](https://www.nve.no/), and is supported by the [Norwegain Environment Agency](http://www.miljodirektoratet.no/en/). The code available in this repository was produced by NIVA.

## Background and code development

The links below provide background to the project and develop code for the new project workflow.

 1. **[Initial data exploration](http://nbviewer.jupyter.org/github/JamesSample/rid/blob/master/notebooks/rid_data_exploration.ipynb)**. Getting to know the project and exploring the results from previous years
 
 2. **[Updating discharge datasets](http://nbviewer.jupyter.org/github/JamesSample/rid/blob/master/notebooks/update_flow_datasets.ipynb)**. Adding new observed and modelled flow datasets from [NVE](https://www.nve.no/)
 
 3. **[Estimating loads at "monitored locations"](http://nbviewer.jupyter.org/github/JamesSample/rid/blob/master/notebooks/estimate_loads.ipynb)**. 155 sites in the Norwegian component of the programme have at least some water chemistry measurements. This notebook estimates pollutant loads for these locations, using a combination of observed flows & discharges, modelled flows and regression analysis. Results for 2015 are estimated using the new code and compared to those reported previously

 4. **[Preparing input files for TEOTIL and running the model](http://nbviewer.jupyter.org/github/JamesSample/rid/blob/master/notebooks/prepare_teotil_inputs.ipynb)**. For parts of Norway where monitoring data are unavailable, the model [TEOTIL](https://brage.bibsys.no/xmlui/handle/11250/214825) is used to estimate nutrient loads (nitrogen and phosphorus). This notebook describes the data processing required to generate input files for the model and compares the output from the new workflow with previous results.
 
 5. **[Summary tables in Microsoft Word](http://nbviewer.jupyter.org/github/JamesSample/rid/blob/master/notebooks/word_data_tables.ipynb)**. Preparing key data tables for the report

 6. **[A new export-coefficient-based pollutant model for Norway](http://nbviewer.jupyter.org/github/JamesSample/rid/blob/master/notebooks/nope_model.ipynb)**. Some changes are required to the way loads are modelled in the RID project. One option is to modify TEOTIL, another is to develop a new model entirely. This notebook develops and tests a simple export-coefficient-based pollutant model (provisionally called NOPE) for simulating Norwegian river loads. 

## Annual data processing

Links to the annual data processing are provided below.

## 2016/17

 1. **[Data processing for "monitored" locations](http://nbviewer.jupyter.org/github/JamesSample/rid/blob/master/notebooks/rid_working_2016-17.ipynb)**. The code developed above is applied using data for 2016 for the 155 sites where water chemistry is measured.
 
 2. **[Updating OSPAR discharge summaries](http://nbviewer.jupyter.org/github/JamesSample/rid/blob/master/notebooks/recalculate_ospar_flows.ipynb)**. Some revisions are required to discharge data previously submitted to OSPAR. This notebook applies the corrections and fills-in new OSPAR templates for the period from 1990 to 2016.
