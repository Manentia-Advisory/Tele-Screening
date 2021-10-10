import React from 'react'
import List from './list.component';
import DISEASE_LIST from '../assets/diseaseList'
import Remark from './remark.component';

import UploadedImage from '../assets/uploaded-Image.jpeg'
import ColorSegmentationImage from '../assets/color-segmentation-image.png'
import BoxSegmentationImage from '../assets/box-segmentation-image.png'

class Result extends React.Component {

    constructor(props) {
        super(props);

        this.state = {
            diseaseList: DISEASE_LIST,
            remarks: ''
        }
    }

    render() {

        const diseases = this.state.diseaseList;

        const handleChangeRemarks = e => {
            this.setState({remarks: e.target.value})
        }

        const handleClick = e => {
            alert("This feature is under development!")
        }

        return (
            <div className="flex-grow w-full max-w-7xl mx-auto xl:pl-8 lg:flex">
            <div className="flex-1 min-w-0 bg-white lg:flex">

            {/* style={{minHeight: "12rem"}} */}
                <div className="lg:w-1/4 lg:border-r">
                    <div className="h-full pl-4 pr-6 py-6 sm:pl-6 lg:pl-8 xl:pl-0">
                        <div className="h-full relative flex flex-col justify-between">
                            {/* <div className="flex flex-col justify-between"> */}
                            <div className="rounded-lg flex flex-col sm:flex-row lg:flex-col content-between">
                                <div className="my-4">
                                    <h2 className="text-lg ml-1 font-medium text-gray-900 mb-1"><span
                                            className="text-gray-500">ID: </span>12399203534</h2>
                                    <img className="mx-auto rounded-md h-80" src={ UploadedImage } alt='' />
                                </div>
                                <div className="lg:w-auto lg:m-0 my-auto mx-auto sm:ml-10 w-11/12">
                                    <h3 className="font-medium text-gray-900">Patient Information</h3>
                                    <dl className="mt-2 border-t border-b border-gray-200 divide-y divide-gray-200">
                                        <div className="py-3 flex justify-between text-sm font-medium">
                                            <dt className="text-gray-500">Name</dt>
                                            <dd className="text-gray-900">Marigot Foster</dd>
                                        </div>

                                        <div className="py-3 flex justify-between text-sm font-medium">
                                            <dt className="text-gray-500">Age</dt>
                                            <dd className="text-gray-900">36 Years</dd>
                                        </div>

                                        <div className="py-3 flex justify-between text-sm font-medium">
                                            <dt className="text-gray-500">Weight</dt>
                                            <dd className="text-gray-900">87 Kg</dd>
                                        </div>

                                        <div className="py-3 flex justify-between text-sm font-medium">
                                            <dt className="text-gray-500">Gender</dt>
                                            <dd className="text-gray-900">Male</dd>
                                        </div>
                                    </dl>
                                </div>
                            </div>
                            <div className="mt-4 mx-4 md:mx-0 md:mt-0 hidden lg:flex lg:flex-col justify-between">
                            <button type="button" onClick={handleClick}
                                className="w-9/12 mr-1 lg:mr-0 lg:w-full mt-2  py-2 px-4 border border-blue-700 rounded-md shadow-sm text-sm font-medium text-blue-800 hover:text-white hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
                                    Edit Marking
                            </button>
                            <button type="button"
                                className="w-9/12 ml-1 lg:ml-0 lg:w-full mt-2 bg-blue-700 py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
                                    Save
                            </button>
                        </div>
                        {/* </div> */}

                        
                        </div>
                        {/* <!-- End left column area --> */}
                    </div>
                </div>

                <div className="bg-white lg:min-w-0 lg:flex-1">
                    <div className="h-full py-6 px-4 sm:px-6 lg:px-8">
                        {/* <!-- Start main area--> */}
                        <div className="relative h-full" style={{minHeight: "36rem"}}>
                            <div className=" inset-0 border border-gray-200 rounded-md">
                                <div className="flex-shrink-0 bg-white">
                                </div>
                                <div className="my-10 mx-5 flex flex-col sm:flex-row justify-between">
                                    <div className="border border-gray-200 rounded-md mx-auto">
                                        <img className="rounded-t-md h-80"
                                            src={ ColorSegmentationImage } alt='' />
                                        <h2 className="text-md text-center my-2 mx-auto font-medium text-gray-900">Lung
                                            Segmentation Image</h2>
                                    </div>
                                    <div className="border border-gray-200 rounded-md mx-auto mt-6 sm:mt-0">
                                        <img className="rounded-t-md h-80"
                                            src={ BoxSegmentationImage } alt='' />
                                        <h2 className="text-md text-center my-2 mx-auto font-medium text-gray-900">
                                            Color/Server Image</h2>
                                    </div>
                                </div>
                                
                                <List list={diseases}/>
                                <Remark value={this.state.remarks} handleChange={handleChangeRemarks}/>

                                <div className="mx-12 mb-8 lg:hidden flex lg:flex-col justify-between">
                                    <button type="button" onClick={handleClick}
                                        className="w-9/12 mr-1.5 lg:mr-0 lg:w-full mt-2  py-2 px-4 border border-blue-700 rounded-md shadow-sm text-sm font-medium text-blue-800 hover:text-white hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
                                            Edit Marking
                                    </button>
                                    <button type="button"
                                        className="w-9/12 ml-1.5 lg:ml-0 lg:w-full mt-2 bg-blue-700 py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
                                            Save
                                    </button>
                                </div>

                                <div className="">
                                    <a href="/" className="block bg-gray-100 text-sm font-medium text-gray-500 text-center px-4 py-4 hover:text-blue-700 hover:bg-gray-200 sm:rounded-b-lg">
                                        Generate Report
                                    </a>
                                </div>
                            </div>
                        </div>
                        {/* <!-- End main area --> */}
                    </div>
                </div>

            </div>
        </div>
        )
    }
}

export default Result;